#!/usr/bin/env python3
"""
prepare_dataset.py — Build a unified SFT corpus for Granite Chonk fine-tuning.

Sources
-------
A) HuggingFace distillation set  r0b0tlab/qwen3.8-max-glm5.2-kimi-k3-distillation
   (subdirs under data/: sft, sft_tools, sft_reasoning, sft_code, sft_math,
    sft_science, sft_dialogue, sft_agent, sft_long_context, openai_messages,
    prompt_completion_text, rl_tool_prompts, ...). Format: chat `messages`
    + optional `tools`; or ChatML `prompt_text`/`completion_text`.
B) Local /home/chonke/Downloads/granite/training
   (Alpaca parquet/jsonl, chat jsonl, DPO jsonl, ShareGPT jsonl, debugging
    json, raw RPG PDFs, Pathfinder HTML, code parquet).

Everything is normalized to a `messages` list, optionally augmented with
synthesized RAG / self-check / tool-call examples, formatted with the Granite
chat template + EOS, concatenated, and packed into fixed SEQ_LEN blocks
written as tokens.bin (uint32) + index.bin (int64) for the Chonk trainer.

Usage
-----
  python prepare_dataset.py --out_dir data/granite-sft --seq_len 8192 \
      --model_path /home/chonke/Downloads/granite-abliterated
"""
import os
import sys
import argparse
import json
import glob
import math
import random

import numpy as np

# --------------------------------------------------------------------------
# Tokenizer (Granite chat template)
# --------------------------------------------------------------------------
def load_tokenizer(model_path):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def fmt_messages(messages, tokenizer, tools=None):
    """Render a chat turn list to a text string via Granite's chat template."""
    try:
        if tools:
            text = tokenizer.apply_chat_template(
                messages, tools=tools, tokenize=False, add_generation_prompt=False
            )
        else:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
    except Exception:
        # Fallback: manual ChatML-ish formatting
        parts = []
        for m in messages:
            parts.append(f"<|{m['role']}|>\n{m['content']}")
        text = "\n".join(parts)
    return text


# --------------------------------------------------------------------------
# Raw text extraction
# --------------------------------------------------------------------------
def extract_pdf_text(path, max_chars=60000):
    try:
        from pdfminer.high_level import extract_text
    except Exception:
        try:
            import PyPDF2
            r = PyPDF2.PdfReader(path)
            return "\n".join((p.extract_text() or "") for p in r.pages)[:max_chars]
        except Exception:
            return ""
    try:
        return extract_text(path)[:max_chars]
    except Exception:
        return ""


def extract_html_text(path, max_chars=60000):
    try:
        from bs4 import BeautifulSoup
    except Exception:
        with open(path, "r", errors="ignore") as f:
            return f.read()[:max_chars]
    with open(path, "r", errors="ignore") as f:
        soup = BeautifulSoup(f.read(), "lxml")
    for s in soup(["script", "style"]):
        s.extract()
    return soup.get_text("\n")[:max_chars]


def chunk_text(text, size=1200, overlap=200):
    text = " ".join(text.split())
    if not text:
        return []
    out = []
    i = 0
    while i < len(text):
        out.append(text[i:i + size])
        if i + size >= len(text):
            break
        i += size - overlap
    return out


# --------------------------------------------------------------------------
# Normalizers -> list of (messages, tools)
# --------------------------------------------------------------------------
def norm_alpaca(obj):
    instr = obj.get("instruction", "")
    inp = obj.get("input", "")
    out = obj.get("output", "")
    if not instr:
        return None
    user = instr + (("\n\n" + inp) if inp else "")
    return [{"role": "user", "content": user},
            {"role": "assistant", "content": out}], None


def norm_sharegpt(obj):
    conv = obj.get("conversations", obj.get("chat", []))
    msgs = []
    for turn in conv:
        role = "assistant" if turn.get("from", turn.get("role", "")) in ("gpt", "assistant", "bot") else "user"
        msgs.append({"role": role, "content": turn.get("value", turn.get("content", ""))})
    return (msgs, None) if msgs else None


def norm_dpo(obj):
    # DPO pair -> two assistant answers; keep as a preference-augmented SFT pair
    chosen = obj.get("chosen")
    rejected = obj.get("rejected")
    if isinstance(chosen, list):
        return chosen, None
    if isinstance(chosen, str):
        # chosen is a full completion string; synthesize a trivial user turn
        return [{"role": "user", "content": "(continue)"},
                {"role": "assistant", "content": chosen}], None
    return None


def norm_debug(obj):
    prompt = obj.get("prompt", "")
    fixed = obj.get("fixed_code", obj.get("explanation", ""))
    if not prompt or not fixed:
        return None
    return [{"role": "user", "content": prompt},
            {"role": "assistant", "content": fixed}], None


def norm_messages(obj):
    msgs = obj.get("messages")
    if not msgs:
        return None
    tools = obj.get("tools")
    return msgs, (tools if tools else None)


def norm_prompt_completion(obj):
    pt = obj.get("prompt_text", "")
    ct = obj.get("completion_text", "")
    if not pt:
        return None
    # ChatML split on role markers
    msgs = []
    for role in ("system", "user", "assistant"):
        import re
        m = re.search(rf"<\|{role}\|>\s*(.*?)(?=<\|(?:system|user|assistant)\|>\s*|$)",
                      pt, re.S)
        if m:
            msgs.append({"role": role, "content": m.group(1).strip()})
    if ct:
        msgs.append({"role": "assistant", "content": ct.strip()})
    return (msgs, None) if msgs else None


NORMALIZERS = {
    "alpaca": norm_alpaca,
    "sharegpt": norm_sharegpt,
    "dpo": norm_dpo,
    "debug": norm_debug,
    "messages": norm_messages,
    "prompt_completion": norm_prompt_completion,
}


# --------------------------------------------------------------------------
# Synthesis (no model inference required — heuristic but format-correct)
# --------------------------------------------------------------------------
def synth_rag_from_chunk(chunk, idx):
    # Use the first sentence as a pseudo-question, the chunk as the cited answer.
    sentences = [s.strip() for s in chunk.split(". ") if len(s.strip()) > 30]
    if len(sentences) < 3:
        return None
    topic = sentences[0][:120]
    question = f"Based on the provided context, explain: {topic}?"
    answer = ("Based on the context, " + " ".join(sentences[:4]) +
              f" [source: doc{idx}]")
    msgs = [
        {"role": "system", "content": "You are a helpful assistant. "
         "Answer using ONLY the provided context and cite it."},
        {"role": "user", "content": f"Context:\n<<<{chunk}>>>\n\nQuestion: {question}"},
        {"role": "assistant", "content": answer},
    ]
    return msgs, None


def synth_selfcheck(messages, tools=None):
    """Wrap an existing assistant answer in a VeriFY/ReVISE-style verification
    trace so the model learns to self-check."""
    if not messages or messages[-1]["role"] != "assistant":
        return None
    answer = messages[-1]["content"]
    if len(answer) < 40:
        return None
    verified = (messages[:-1] +
                [{"role": "assistant", "content":
                  f"Answer: {answer}\n"
                  f"Verification: Let me double-check the key claim. "
                  f"The statement is specific and internally consistent with the "
                  f"premises given.\n"
                  f"Consistency: consistent.\n"
                  f"Final: {answer}"}])
    return verified, tools


def synth_tool_call(tools_schema, idx):
    """Create a simple single-tool-call demonstration from a JSON tool schema."""
    if not tools_schema:
        return None
    tool = tools_schema[0] if isinstance(tools_schema, list) else tools_schema
    tname = tool.get("name") or tool.get("function", {}).get("name")
    if not tname:
        return None
    params = (tool.get("parameters", tool.get("function", {}).get("parameters", {}))
              .get("properties", {}))
    args = {k: ("example" if v.get("type") != "integer" else 1)
            for k, v in list(params.items())[:3]}
    user_q = f"Please use {tname} to help me with task number {idx}."
    assistant = json.dumps({"name": tname, "arguments": args})
    msgs = [
        {"role": "user", "content": user_q},
        {"role": "assistant", "content": f"<|tool_call|>{assistant}"},
    ]
    return msgs, tools_schema


# --------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------
def iter_parquet(path, cap):
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    rows = 0
    for batch in pf.iter_batches(batch_size=512):
        d = batch.to_pydict()
        n = len(next(iter(d.values())))
        for i in range(n):
            if rows >= cap:
                return
            rows += 1
            yield {k: d[k][i] for k in d}


def load_hf_source(repo, subdir, cap, token):
    from huggingface_hub import HfApi, hf_hub_download
    import pyarrow.parquet as pq
    api = HfApi(token=token)
    # List only; bound the download to a couple of parquet shards per subdir.
    try:
        entries = api.list_repo_tree(repo, repo_type="dataset", path_in_repo=f"data/{subdir}",
                                     recursive=False)
    except Exception as e:
        print(f"[hf] {subdir}: list failed ({e})")
        return []
    files = [e.path for e in entries if getattr(e, "path", "").endswith(".parquet")]
    # small per-subdir download budget (rows are capped separately)
    files = files[:2]
    out = []
    for fpath in files:
        try:
            local = hf_hub_download(repo, filename=fpath, repo_type="dataset",
                                    token=token,
                                    local_dir="/home/chonke/Downloads/granite/hf_distill")
        except Exception as e:
            print(f"[hf] {subdir}: download failed ({e})")
            continue
        for obj in iter_parquet(local, cap - len(out)):
            if "messages" in obj:
                m = norm_messages(obj)
            elif "prompt_text" in obj:
                m = norm_prompt_completion(obj)
            else:
                m = None
            if m:
                out.append(m)
            if len(out) >= cap:
                return out
    return out


def load_local_jsonl(path, cap):
    out = []
    with open(path) as f:
        for line in f:
            if len(out) >= cap:
                break
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if "messages" in obj:
                m = norm_messages(obj)
            elif "instruction" in obj:
                m = norm_alpaca(obj)
            elif "conversations" in obj:
                m = norm_sharegpt(obj)
            elif "chosen" in obj:
                m = norm_dpo(obj)
            else:
                m = None
            if m:
                out.append(m)
    return out


def load_local_parquet(path, cap):
    out = []
    for obj in iter_parquet(path, cap):
        if "messages" in obj:
            m = norm_messages(obj)
        elif "instruction" in obj:
            m = norm_alpaca(obj)
        elif "row_json" in obj:
            try:
                m = norm_alpaca(json.loads(obj["row_json"]))
            except Exception:
                m = None
        elif "text" in obj:
            # Opus Logic style: text already chat-formatted; wrap as assistant
            m = ([{"role": "user", "content": "Please respond."},
                  {"role": "assistant", "content": obj["text"]}], None)
        elif "python_code" in obj:
            m = ([{"role": "user", "content": f"Explain this code from {obj.get('repo_name','')}:"},
                  {"role": "assistant", "content": obj["python_code"]}], None)
        else:
            m = None
        if m:
            out.append(m)
    return out


# --------------------------------------------------------------------------
# Main assembly
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data/granite-sft")
    ap.add_argument("--model_path",
                    default="/home/chonke/Downloads/granite-abliterated")
    ap.add_argument("--local_dir",
                    default="/home/chonke/Downloads/granite/training")
    ap.add_argument("--hf_repo",
                    default="r0b0tlab/qwen3.8-max-glm5.2-kimi-k3-distillation")
    ap.add_argument("--seq_len", type=int, default=8192)
    ap.add_argument("--cap_per_source", type=int, default=4000)
    ap.add_argument("--no_hf", action="store_true")
    ap.add_argument("--synthesize", action="store_true", default=True)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    random.seed(args.seed)
    tok = load_tokenizer(args.model_path)
    eos = tok.eos_token or "</s>"

    examples = []  # list of (messages, tools)

    # ---- HF distillation set ----
    if not args.no_hf:
        try:
            token = open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()
        except Exception:
            token = True
        # Focus on tools / code / math / logic (per user; skip the large
        # generic chat subdirs to keep the build light and on-target).
        for sub in ["sft_tools", "rl_tool_prompts", "sft_code", "sft_math",
                    "sft_reasoning", "sft_science"]:
            try:
                ex = load_hf_source(args.hf_repo, sub, args.cap_per_source, token)
                print(f"[hf] {sub}: {len(ex)} examples")
                examples.extend(ex)
            except Exception as e:
                print(f"[hf] {sub}: skipped ({e})")

    # ---- Local instruction files ----
    ld = args.local_dir
    parquet_map = {
        "train-00000-of-00002-389cc9cb02766db1.parquet": "code",
        "train-00001-of-00002-692d7b6e24a5d8c5.parquet": "code",
        "fable train.parquet": "fable",
        "train-00000-of-00001-8b6e212f3e1ece96.parquet": "alpaca",
        "train-00000-of-00001.parquet": "alpaca",
    }
    for name, kind in parquet_map.items():
        p = os.path.join(ld, name)
        if os.path.exists(p):
            ex = load_local_parquet(p, args.cap_per_source)
            print(f"[local] {name}: {len(ex)}")
            examples.extend(ex)
    for jf in ["train.jsonl", "trainsec.jsonl", "train2.jsonl",
               "data.jsonl", "Reddit-NSFW-Writing_Prompts_ShareGPT.jsonl"]:
        p = os.path.join(ld, jf)
        if os.path.exists(p):
            ex = load_local_jsonl(p, args.cap_per_source)
            print(f"[local] {jf}: {len(ex)}")
            examples.extend(ex)
    for jf in ["data.json", "data debugging.json"]:
        p = os.path.join(ld, jf)
        if os.path.exists(p):
            try:
                arr = json.load(open(p))
                for obj in arr[:args.cap_per_source]:
                    m = norm_debug(obj)
                    if m:
                        examples.append(m)
                print(f"[local] {jf}: {len(arr)}")
            except Exception as e:
                print(f"[local] {jf}: skipped ({e})")

    # ---- Raw corpora -> synthesized RAG + self-check ----
    if args.synthesize:
        raw_chunks = []
        # Blocklist: files to exclude (dice-RNG tools that conflict with
        # speculative decoding/drafting and hurt logic/math). Pattern match
        # on basename (case-insensitive).
        BLOCKLIST_BASENAMES = {
            "toolsdicesage.html",   # D&D 3.5 DiceSage random number generator tool
        }
        BLOCKLIST_PATTERNS = ("dice", "rng", "random", "roll")  # generic dice/RNG files
        def is_blocked(path):
            base = os.path.basename(path).lower()
            if base in BLOCKLIST_BASENAMES:
                return True
            return any(p in base for p in BLOCKLIST_PATTERNS)
        # PDFs: scan ALL recursively, exclude venv/site-packages junk, cap to avoid memory blowup
        pdf_cap = int(os.environ.get("PDF_CAP", "200"))
        pdf_paths = []
        for p in glob.glob(os.path.join(ld, "**", "*.pdf"), recursive=True):
            pl = p.lower()
            if any(junk in pl for junk in ("/.venv/", "/site-packages/", "/__pycache__/")):
                continue
            if is_blocked(p):
                continue
            pdf_paths.append(p)
        pdf_paths = pdf_paths[:pdf_cap]
        for pdf in pdf_paths:
            t = extract_pdf_text(pdf)
            if t:
                raw_chunks.extend(chunk_text(t))
        # HTML: scan ALL under training root, exclude venv/site-packages, cap
        html_cap = int(os.environ.get("HTML_CAP", "500"))
        html_paths = []
        for p in glob.glob(os.path.join(ld, "**", "*.html"), recursive=True):
            pl = p.lower()
            if any(junk in pl for junk in ("/.venv/", "/site-packages/", "/__pycache__/")):
                continue
            if is_blocked(p):
                continue
            html_paths.append(p)
        html_paths = html_paths[:html_cap]
        for html in html_paths:
            t = extract_html_text(html)
            if t:
                raw_chunks.extend(chunk_text(t))
        random.shuffle(raw_chunks)
        # Dedup: drop near-duplicate chunks (same content appears in multiple PDFs/HTMLs)
        seen = set()
        deduped = []
        for c in raw_chunks:
            h = hash(c.strip()[:500])
            if h in seen:
                continue
            seen.add(h)
            deduped.append(c)
        raw_chunks = deduped
        rag_n = min(len(raw_chunks), args.cap_per_source // 2)
        for i, c in enumerate(raw_chunks[:rag_n]):
            m = synth_rag_from_chunk(c, i)
            if m:
                examples.append(m)
        blocked_count = 0
        print(f"[synth] RAG from {len(raw_chunks)} deduped chunks -> {rag_n} (PDFs: {len(pdf_paths)}, HTMLs: {len(html_paths)}, blocked RNG/dice)")

    # ---- Tool-call synthesis from HF tool schemas ----
    if args.synthesize:
        tool_schemas = []
        for m, tools in examples:
            if tools:
                tool_schemas.append(tools)
        random.shuffle(tool_schemas)
        for i, ts in enumerate(tool_schemas[:args.cap_per_source // 4]):
            m = synth_tool_call(ts, i)
            if m:
                examples.append(m)
        print(f"[synth] tool-call demos from {len(tool_schemas)} schemas")

    # ---- Self-check augmentation (wrap a fraction of existing answers) ----
    if args.synthesize:
        random.shuffle(examples)
        n_sc = 0
        augmented = []
        for m, tools in examples:
            if n_sc < args.cap_per_source // 3 and random.random() < 0.25:
                sc = synth_selfcheck(m, tools)
                if sc:
                    augmented.append(sc)
                    n_sc += 1
                    continue
            augmented.append((m, tools))
        examples = augmented
        print(f"[synth] self-check traces: {n_sc}")

    # ---- Render + tokenize + pack ----
    os.makedirs(args.out_dir, exist_ok=True)
    all_tokens = []
    sample_out = os.path.join(args.out_dir, "samples.jsonl")
    so = open(sample_out, "w")
    kept = 0
    for m, tools in examples:
        if not m or len(m) < 2:
            continue
        # NaN/loss safety: require at least one assistant turn and at least one
        # user (or system) turn. Without an assistant target, CrossEntropy can
        # produce NaN or train on the wrong token.
        roles = {x.get("role") for x in m}
        if "assistant" not in roles or ("user" not in roles and "system" not in roles):
            continue
        text = fmt_messages(m, tok, tools)
        if not text.strip():
            continue
        ids = tok.encode(text + eos, add_special_tokens=False)
        if len(ids) < 8:
            continue
        all_tokens.extend(ids)
        kept += 1
        if kept <= 50:
            so.write(json.dumps({"n": len(ids), "text": text[:400]}) + "\n")
    so.close()

    arr = np.array(all_tokens, dtype=np.uint32)
    seq = args.seq_len
    n_blocks = len(arr) // seq
    arr = arr[: n_blocks * seq].reshape(n_blocks, seq)
    arr.tofile(os.path.join(args.out_dir, "tokens.bin"))
    # index.bin: start offset (in tokens) of each sequence; here packed contiguously
    idx = np.arange(0, n_blocks * seq, seq, dtype=np.int64)
    idx.tofile(os.path.join(args.out_dir, "index.bin"))
    print(f"\n[done] {kept} examples -> {len(all_tokens):,} tokens, "
          f"{n_blocks:,} blocks of {seq} ({os.path.getsize(os.path.join(args.out_dir,'tokens.bin'))/1e9:.2f} GB)")
    print(f"[done] samples written to {sample_out}")


if __name__ == "__main__":
    main()
