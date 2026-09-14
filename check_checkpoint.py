"""Check checkpoint key structure."""
import safetensors
import os

d = r'D:\models\mineru_models\models\OpenDataLab--PDF-Extract-Kit-1.0\snapshots\master\models\Layout\PP-DocLayoutV2'
f = 'model.safetensors'
st = safetensors.safe_open(os.path.join(d, f), 'pt')
keys = [k for k in st.keys() if k.startswith('model.encoder.')]

print(f'Total model.encoder.* keys: {len(keys)}')
print(f'  encoder.encoder.*: {sum(1 for k in keys if k.startswith("model.encoder.encoder."))}')
print(f'  encoder.aifi.*: {sum(1 for k in keys if k.startswith("model.encoder.aifi."))}')
print(f'  encoder.enc_layer.*: {sum(1 for k in keys if k.startswith("model.encoder.enc_layer"))}')
print(f'  encoder.encode_proj.*: {sum(1 for k in keys if k.startswith("model.encoder.encode_proj"))}')
print(f'  encoder.lateral_convs.*: {sum(1 for k in keys if k.startswith("model.encoder.lateral_convs"))}')
print(f'  encoder.pan_blocks.*: {sum(1 for k in keys if k.startswith("model.encoder.pan_blocks"))}')
print(f'  encoder.decoder_layers.*: {sum(1 for k in keys if k.startswith("model.encoder.decoder_layers"))}')
print()

# Also check model structure vs checkpoint
# Check all top-level keys under model.encoder
prefixes = set()
for k in keys:
    parts = k.split('.')
    if len(parts) >= 3:
        prefixes.add('.'.join(parts[:3]))
    elif len(parts) >= 2:
        prefixes.add('.'.join(parts[:2]))

print('Checkpoint submodules under model.encoder:')
for p in sorted(prefixes):
    count = sum(1 for k in keys if k.startswith(p))
    print(f'  {p}: {count} keys')

print()
print('=== model.encoder.encoder keys ===')
for k in keys:
    if k.startswith('model.encoder.encoder.'):
        print(f'  {k}')