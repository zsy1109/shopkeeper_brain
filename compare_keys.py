"""Compare model state dict with fixed checkpoint to find missing/unexpected keys."""
import safetensors.torch
import os, gc

os.environ.setdefault('MINERU_MODEL_DIR', r'D:\PyCharm项目\shopkeeper_brain_1\models')
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

import torch
torch.backends.cudnn.benchmark = False

from mineru.model.layout.pp_doclayoutv2 import PPDocLayoutV2Config, PPDocLayoutV2ForObjectDetection

model_dir = r'D:\models\mineru_models\models\OpenDataLab--PDF-Extract-Kit-1.0\snapshots\master\models\Layout\PP-DocLayoutV2'

config = PPDocLayoutV2Config.from_pretrained(model_dir)
model = PPDocLayoutV2ForObjectDetection.from_pretrained(model_dir, config=config)

model_state = model.state_dict()
model_keys = set(model_state.keys())

fixed_path = os.path.join(model_dir, 'model.safetensors.fixed')
fixed_state = safetensors.torch.load_file(fixed_path)
fixed_keys = set(fixed_state.keys())

missing = model_keys - fixed_keys
unexpected = fixed_keys - model_keys

print(f'Model keys: {len(model_keys)}')
print(f'Fixed keys: {len(fixed_keys)}')
print(f'MISSING ({len(missing)}):')
for k in sorted(missing):
    print(f'  {k}')
print(f'\nUNEXPECTED ({len(unexpected)}):')
for k in sorted(unexpected):
    print(f'  {k}')

# Also check original checkpoint to understand the mapping
orig_path = os.path.join(model_dir, 'model.safetensors.orig')
if os.path.isfile(orig_path):
    orig_state = safetensors.torch.load_file(orig_path)
    orig_keys = set(orig_state.keys())
    orig_missing = model_keys - orig_keys
    orig_unexpected = orig_keys - model_keys
    print(f'\nOriginal checkpoint:')
    print(f'  Orig unexpected ({len(orig_unexpected)}):')
    for k in sorted(orig_unexpected):
        print(f'    {k}')

del model, model_state, fixed_state
gc.collect()
torch.cuda.empty_cache()