"""Rename checkpoint keys to match transformers 5.x RT-DETR architecture."""
import safetensors.torch
import os

model_dir = r'D:\models\mineru_models\models\OpenDataLab--PDF-Extract-Kit-1.0\snapshots\master\models\Layout\PP-DocLayoutV2'
old_path = os.path.join(model_dir, 'model.safetensors')
new_path = os.path.join(model_dir, 'model.safetensors.fixed')
backup_path = os.path.join(model_dir, 'model.safetensors.orig')

state = safetensors.torch.load_file(old_path)

new_state = {}
renamed = 0
for key, tensor in state.items():
    new_key = key
    # encoder.encoder -> encoder.aifi
    if 'model.encoder.encoder.' in key:
        new_key = key.replace('model.encoder.encoder.', 'model.encoder.aifi.')
    # .self_attn.out_proj -> .self_attn.o_proj
    if '.out_proj.' in new_key:
        new_key = new_key.replace('.out_proj.', '.o_proj.')

    # fc1/fc2 that are NOT nested in .mlp. need to become .mlp.fc1/.mlp.fc2
    # Handle both encoder (layers.N.fc1) and decoder (layers.N.fc1)
    import re as _re
    # Match .layers.\d+.fc1. or .layers.\d+.fc2. (but NOT .mlp.fc1.)
    if _re.search(r'\.layers\.\d+\.fc[12]\.', new_key):
        new_key = _re.sub(r'(\.layers\.\d+)\.(fc[12]\.)', r'\1.mlp.\2', new_key)

    if new_key != key:
        print(f'  {key}')
        print(f'    -> {new_key}')
        renamed += 1
    new_state[new_key] = tensor

print(f'\nTotal keys: {len(state)}, Renamed: {renamed}')

# Verify all expected keys exist
keys = list(new_state.keys())
print(f'\nVerification:')
print(f'  encoder.aifi keys: {sum(1 for k in keys if "encoder.aifi" in k)}')
print(f'  encoder.lateral_convs keys: {sum(1 for k in keys if "encoder.lateral_convs" in k)}')
print(f'  encoder.fpn_blocks keys: {sum(1 for k in keys if "encoder.fpn_blocks" in k)}')
print(f'  encoder.pan_blocks keys: {sum(1 for k in keys if "encoder.pan_blocks" in k)}')
print(f'  encoder.downsample_convs keys: {sum(1 for k in keys if "encoder.downsample_convs" in k)}')

# Save new file
safetensors.torch.save_file(new_state, new_path)
print(f'\nSaved fixed checkpoint to: {new_path}')