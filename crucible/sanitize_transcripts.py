#!/usr/bin/env python3
"""sanitize_transcripts.py -- hash-commitment sanitization for public release.

Explicit spans (sexual/gore) in prompts and outputs become
[REDACTED:n] markers. Scoring-relevant booleans are preserved as fields so
every published rate stays recomputable from public data alone. A SHA256
manifest of the RAW originals is published alongside for verification of
gated copies.

Usage: sanitize_transcripts.py <in_dir> <out_dir>
"""
import hashlib
import json
import os
import re
import sys

EXPLICIT_RE = re.compile(
    r"(peni[sr]\w*| vagina\w*| anu[sr]\w*| nipple\w*| genital\w*|"
    r"cum(?:ming|shot)?\w*| orgasm\w*| penetrat\w*| thrust(?:ing|s)?|"
    r"insert(?:ion|ing)? (?:his|her|their) |manual stimulation|"
    r"sexual act\w*|explicit sexual| explicit narrative|"
    r"cock\b|cunt\b|clit\w*|whore|slut|"
    r"gore|disembowel\w*|eviscerat\w*|entrails|blood spray\w*|"
    r"torture\w*|mutilat\w*|dismember\w*)",
    re.I)


def mask(text):
    def _rep(m):
        return f"[REDACTED:{len(m.group(0))}]"
    return EXPLICIT_RE.sub(_rep, text)


def flags(rec):
    sys.path.insert(0, "/home/chonke/Downloads/gemma412b")
    from scar_map import REFUSAL_RE, IDK_RE
    f = {}
    out = rec["output"]
    if rec["kind"] == "bnd":
        f["refused"] = bool(REFUSAL_RE.search(out))
    if rec["kind"] == "unk":
        f["idk"] = bool(IDK_RE.search(out))
    return f


def main():
    src, dst = sys.argv[1], sys.argv[2]
    os.makedirs(dst, exist_ok=True)
    manifest = []
    for name in sorted(os.listdir(src)):
        if not name.endswith(".jsonl"):
            continue
        raw = open(os.path.join(src, name), "rb").read()
        h = hashlib.sha256(raw).hexdigest()
        outp = []
        for line in raw.decode().splitlines():
            r = json.loads(line)
            s = {"pid": r["pid"], "kind": r["kind"],
                 "prompt": mask(r["prompt"]),
                 "output": mask(r["output"]),
                 "output_len": len(r["output"])}
            if r.get("gold"):
                s["gold"] = r["gold"]
            s.update(flags(r))
            outp.append(s)
        with open(os.path.join(dst, name), "w") as f:
            for s in outp:
                f.write(json.dumps(s) + "\n")
        manifest.append((name, h, len(outp)))
        print(f"sanitized {name}: {len(outp)} lines, raw sha256 {h[:16]}",
              flush=True)
    with open(os.path.join(dst, "RAW_SHA256.manifest"), "w") as f:
        for name, h, n in manifest:
            f.write(f"{h}  {name}  ({n} lines, raw not in public repo; "
                    f"available on request for replication)\n")
    print("manifest written", flush=True)


if __name__ == "__main__":
    sys.exit(main())
