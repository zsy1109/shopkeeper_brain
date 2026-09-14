"""Test MinerU PDF parsing with the fixed checkpoint."""
import os, sys, time

os.environ.setdefault('MINERU_MODEL_DIR', r'D:\models\mineru_models\models\OpenDataLab--PDF-Extract-Kit-1.0\snapshots\master\models')
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('MINERU_DEVICE_MODE', 'cpu')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

input_pdf = r'D:\PyCharm项目\shopkeeper_brain_1\docs\万用表的使用_origin.pdf'
output_dir = r'D:\PyCharm项目\shopkeeper_brain_1\data\tmp\mineru_test_output'

print(f'Input: {input_pdf}')
print(f'Output: {output_dir}')
print(f'Model dir: {os.environ["MINERU_MODEL_DIR"]}')
print()

args = [
    '-p', input_pdf,
    '-o', output_dir,
    '-b', 'pipeline',
    '-m', 'txt',
    '-l', 'ch',
]

print(f'Running: mineru {" ".join(args)}')
print()

import io, contextlib
import click
from mineru.cli.client import main as mineru_main

output_buf = io.StringIO()
exit_code = 0
t0 = time.time()

try:
    with contextlib.redirect_stdout(output_buf), contextlib.redirect_stderr(output_buf):
        mineru_main(args=args, standalone_mode=False)
except click.exceptions.Exit as e:
    exit_code = e.exit_code
    print(f'Exit code: {e.exit_code}')
except Exception as e:
    import traceback
    traceback.print_exc()
    exit_code = 1

dt = time.time() - t0
print(f'Time: {dt:.1f}s')
print(f'Exit code: {exit_code}')

output_text = output_buf.getvalue()
print(f'\n--- MinerU output ---')
for line in output_text.splitlines()[-30:]:
    print(f'  {line}')

if exit_code == 0:
    md_files = list(os.listdir(output_dir))
    print(f'\nOutput files: {md_files}')
    md_file = os.path.join(output_dir, '万用表的使用_origin.md')
    if os.path.isfile(md_file):
        with open(md_file, 'r', encoding='utf-8') as f:
            md_content = f.read()[:2000]
        print(f'\nMD file content (first 2000 chars):')
        print(md_content)
    else:
        print(f'MD file not found at: {md_file}')
        # Check the output directory structure
        for root, dirs, files in os.walk(output_dir):
            for f in files:
                print(f'  {os.path.join(root, f)}')