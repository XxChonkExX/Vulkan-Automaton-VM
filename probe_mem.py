import sys, time, faulthandler, os
sys.path.insert(0, '/home/chonke/Vulkan-Automaton-VM/python/vulkanvm_torch')
sys.path.insert(0, '/home/chonke/Vulkan-Automaton-VM/_build')
faulthandler.enable()
os.environ['HIP_LAUNCH_BLOCKING'] = '1'
import torch
from transformers import AutoConfig
from chonk import build_lora_chonk_setup, patch_linear_cache_for_chunked_training

torch.manual_seed(0)
config = AutoConfig.from_pretrained('/home/chonke/local_training/models/qwen36_27ablit', trust_remote_code=True)
text_cfg = config.get_text_config(decoder=True)

setup = build_lora_chonk_setup(
    '/home/chonke/local_training/models/qwen36_27ablit', config,
    batch_size=1, max_cache_len=131072, lora_r=64, lora_alpha=128,
    attn_implementation='eager',
)
pool = setup['pool']; kv_cache = setup['kv_cache']; model = setup['model']
model.train()
patch_linear_cache_for_chunked_training()
print('SETUP_DONE', flush=True)

seq = torch.randint(0, text_cfg.vocab_size, (1, 2048), device='cuda')

t0 = time.time()
out = model(input_ids=seq, past_key_values=kv_cache, use_cache=True,
            cache_position=torch.arange(0, 2048, device='cuda'))
torch.cuda.synchronize()
print(f'FWD1_DONE {time.time()-t0:.1f}s', flush=True)

loss = torch.nn.functional.cross_entropy(
    out.logits[:, :-1].reshape(-1, out.logits.shape[-1]),
    seq[:, 1:].reshape(-1))
loss.backward()
torch.cuda.synchronize()
print(f'BWD1_DONE {time.time()-t0:.1f}s', flush=True)
del loss, out

seq2 = torch.randint(0, text_cfg.vocab_size, (1, 2048), device='cuda')
out = model(input_ids=seq2, past_key_values=kv_cache, use_cache=True,
            cache_position=torch.arange(2048, 4096, device='cuda'))
torch.cuda.synchronize()
print(f'FWD2_DONE {time.time()-t0:.1f}s', flush=True)
loss = torch.nn.functional.cross_entropy(
    out.logits[:, :-1].reshape(-1, out.logits.shape[-1]),
    seq2[:, 1:].reshape(-1))
loss.backward()
torch.cuda.synchronize()
print(f'BWD2_DONE {time.time()-t0:.1f}s', flush=True)
print('PROBE2 DONE', flush=True)