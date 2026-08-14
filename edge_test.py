import sys, time, faulthandler, json
sys.path.insert(0, '/home/chonke/Vulkan-Automaton-VM/python/vulkanvm_torch')
sys.path.insert(0, '/home/chonke/Vulkan-Automaton-VM/_build')
faulthandler.enable()
import torch
from transformers import AutoConfig
from chonk import build_lora_chonk_setup, reset_chonk_cache, patch_linear_cache_for_chunked_training

CHUNK = int(sys.argv[1])
SEQ = int(sys.argv[2])
RESULT = sys.argv[3]

torch.manual_seed(0)
config = AutoConfig.from_pretrained('/home/chonke/local_training/models/qwen36_27ablit', trust_remote_code=True)
text_cfg = config.get_text_config(decoder=True)

t_setup = time.time()
setup = build_lora_chonk_setup(
    '/home/chonke/local_training/models/qwen36_27ablit', config,
    batch_size=1, max_cache_len=131072, lora_r=64, lora_alpha=128,
    attn_implementation='eager',
)
pool = setup['pool']; kv_cache = setup['kv_cache']; model = setup['model']
model.train()
patch_linear_cache_for_chunked_training()

seq = torch.randint(0, text_cfg.vocab_size, (1, SEQ), device='cuda')
peaks = []
t0 = time.time()
n_chunks = SEQ // CHUNK
for i in range(n_chunks):
    cs, ce = i * CHUNK, (i + 1) * CHUNK
    torch.cuda.reset_peak_memory_stats()
    out = model(input_ids=seq[:, cs:ce], past_key_values=kv_cache, use_cache=True,
                cache_position=torch.arange(cs, ce, device='cuda'))
    loss = torch.nn.functional.cross_entropy(
        out.logits[:, :-1].reshape(-1, out.logits.shape[-1]),
        seq[:, cs+1:ce].reshape(-1))
    loss.backward()
    torch.cuda.synchronize()
    peaks.append(torch.cuda.max_memory_allocated() / 1e9)
    del loss, out
    if i % 4 == 3:
        torch.cuda.empty_cache()
        print(f'  chunk {i+1}/{n_chunks} ok ({time.time()-t0:.0f}s, hip_peak={peaks[-1]:.1f}GB)', flush=True)

with open(RESULT, 'w') as f:
    json.dump({
        'chunk': CHUNK, 'seq': SEQ, 'status': 'PASS',
        'setup_s': round(t_setup - time.time() + (time.time() - t_setup) + 0, 1),
        'total_s': round(time.time() - t0, 1),
        'hip_peaks_gb': [round(p, 2) for p in peaks],
        'max_hip_peak_gb': round(max(peaks), 2),
        'pool_stats': pool.stats(),
    }, f, indent=1)
print('SWEEP_CONFIG_PASS', flush=True)