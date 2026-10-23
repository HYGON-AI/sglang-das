"""Export WeLM OE embedding tables to /dev/shm (bf16 row-major), once per host.

/dev/shm/welm_oe/<i>.bin holds model.oe_embed.<i>.weight [vocab_i, oe_dim].
Servers started with WELM_OE_HOST_DIR=/dev/shm/welm_oe read rows from there
(host-mapped, see welmv4.HostOEEmbedding) instead of holding the tables in HBM.
"""
import json, os, sys, time
from safetensors import safe_open

model = sys.argv[1] if len(sys.argv) > 1 else '/model/tencent/WeLM-v4.5-80B-A3B-0813-MTP'
out = sys.argv[2] if len(sys.argv) > 2 else '/dev/shm/welm_oe'
os.makedirs(out, exist_ok=True)
idx = json.load(open(os.path.join(model, 'model.safetensors.index.json')))['weight_map']
meta = {}
for i in range(64):
    name = f'model.oe_embed.{i}.weight'
    if name not in idx:
        break
    t = time.time()
    with safe_open(os.path.join(model, idx[name]), 'pt') as f:
        w = f.get_tensor(name)
    path = os.path.join(out, f'{i}.bin')
    tmp = path + '.tmp'
    w.contiguous().view(-1).view(dtype=__import__('torch').int16).numpy().tofile(tmp)
    os.replace(tmp, path)
    meta[i] = {'shape': list(w.shape), 'dtype': str(w.dtype)}
    print(name, list(w.shape), w.dtype, round(time.time() - t, 1), 's', flush=True)
json.dump({'model': model, 'tables': meta}, open(os.path.join(out, 'meta.json'), 'w'))
