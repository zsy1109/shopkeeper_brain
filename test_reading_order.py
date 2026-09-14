"""Test batch_predict internals step by step with CUDA sync."""
import os, sys, time, traceback

os.environ.setdefault('MINERU_MODEL_DIR', r'D:\PyCharm项目\shopkeeper_brain_1\models')
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

import torch
torch.backends.cudnn.benchmark = False

print('=== Init model ===', flush=True)
t0 = time.time()
from mineru.backend.pipeline.model_init import MineruPipelineModel

model = MineruPipelineModel(
    device='cuda',
    lang='ch',
    table_config={'enable': True},
    formula_config={'enable': True},
)
print(f'Model init OK! dt={time.time()-t0:.1f}s', flush=True)

layout_model = model.layout_model

print('\n=== Load PDF pages ===', flush=True)
import pypdfium2 as pdfium
input_pdf = r'D:\PyCharm项目\shopkeeper_brain_1\docs\万用表的使用_origin.pdf'
doc = pdfium.PdfDocument(input_pdf)
n_pages = len(doc)
print(f'PDF pages: {n_pages}', flush=True)

from mineru.utils.pdf_reader import page_to_image

images = []
for i in range(min(3, n_pages)):
    page = doc[i]
    img, scale = page_to_image(page, dpi=200)
    images.append(img)
    print(f'  Page {i}: {img.size}', flush=True)
doc.close()

print('\n=== Step-by-step batch_predict ===', flush=True)

pixel_values_list = []
target_sizes = []
for i, image in enumerate(images):
    pv, ts = layout_model._preprocess_single_image(image)
    pixel_values_list.append(pv)
    target_sizes.append(ts)
    print(f'  Preprocess Page {i}: pv={pv.shape}, target={ts}', flush=True)

print('\nStep 1: stack ->', flush=True)
batch_tensor = torch.stack(pixel_values_list, dim=0).to('cuda')
print(f'  batch_tensor: {batch_tensor.shape}, range=[{batch_tensor.min():.3f}, {batch_tensor.max():.3f}]', flush=True)
torch.cuda.synchronize()
print('  stack OK', flush=True)

print('\nStep 2: model forward ->', flush=True)
with torch.no_grad():
    outputs = layout_model.model(pixel_values=batch_tensor)
torch.cuda.synchronize()
print(f'  outputs OK: pred_boxes={outputs.pred_boxes.shape}, logits={outputs.logits.shape}', flush=True)
if outputs.order_logits is not None:
    print(f'  order_logits={outputs.order_logits.shape}', flush=True)

print('\nStep 3: post_process ->', flush=True)
try:
    predictions = layout_model._post_process_object_detection(outputs, target_sizes)
    torch.cuda.synchronize()
    print(f'  post_process OK: {len(predictions)} preds', flush=True)
except Exception as e:
    torch.cuda.synchronize()
    print(f'  post_process ERROR: {type(e).__name__}: {e}', flush=True)
    traceback.print_exc()

print('\n=== DONE ===', flush=True)