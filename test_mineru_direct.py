"""Direct PDF parse via pipeline API (bypass CLI subprocess)."""
import os, sys, time, traceback
from pathlib import Path

os.environ.setdefault('MINERU_MODEL_DIR', r'D:\PyCharm项目\shopkeeper_brain_1\models')
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

input_pdf = r'D:\PyCharm项目\shopkeeper_brain_1\docs\万用表的使用_origin.pdf'
output_dir = r'D:\PyCharm项目\shopkeeper_brain_1\knowledge\processor\import_processor\temp_dir\test_pipeline_api'

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

print('\n=== Load PDF ===', flush=True)
import fitz  # PyMuPDF
doc = fitz.open(input_pdf)
print(f'PDF pages: {doc.page_count}', flush=True)

print('\n=== Process page 0 ===', flush=True)
page = doc[0]
pix = page.get_pixmap(dpi=200)
img_bytes = pix.tobytes("png")
from PIL import Image
import io
img = Image.open(io.BytesIO(img_bytes))
print(f'Page image size: {img.size}', flush=True)

# Test each model step
print('\n--- Layout ---', flush=True)
t1 = time.time()
try:
    layout_result = model.layout_model.predict(img)
    print(f'Layout OK! dt={time.time()-t1:.1f}s, {len(layout_result)} regions', flush=True)
    for r in layout_result[:5]:
        print(f'  {r["label"]} score={r["score"]} bbox={r["bbox"]}', flush=True)
except Exception as e:
    print(f'Layout ERROR: {e}', flush=True)
    traceback.print_exc()

print('\n--- MFR ---', flush=True)
t1 = time.time()
try:
    mfr_result = model.mfr_model.predict('', img)  # some MFR models need image_path, image
    print(f'MFR OK! dt={time.time()-t1:.1f}s', flush=True)
except Exception as e:
    print(f'MFR predict signature issue, trying batch_predict...', flush=True)
    try:
        mfr_result = model.mfr_model.batch_predict([''], [img])  # alternate API
        print(f'MFR batch_predict OK! dt={time.time()-t1:.1f}s', flush=True)
    except Exception as e2:
        print(f'MFR ERROR: {e2}', flush=True)

print('\n--- OCR ---', flush=True)
t1 = time.time()
try:
    ocr_result = model.ocr_model.ocr(img)
    print(f'OCR OK! dt={time.time()-t1:.1f}s', flush=True)
except Exception as e:
    print(f'OCR ERROR: {e}', flush=True)
    traceback.print_exc()

print('\n=== DONE ===', flush=True)
doc.close()