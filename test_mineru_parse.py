"""Test mineru CLI parsing (patches via sitecustomize.py)."""
import os, sys, time

os.environ.setdefault('MINERU_MODEL_DIR', r'D:\PyCharm项目\shopkeeper_brain_1\models')
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

from mineru.cli.client import main as mineru_main
import click

input_pdf = r'D:\PyCharm项目\shopkeeper_brain_1\docs\万用表的使用_origin.pdf'
output_dir = r'D:\PyCharm项目\shopkeeper_brain_1\knowledge\processor\import_processor\temp_dir\test_cli_output'

args = [
    '-p', input_pdf,
    '-o', output_dir,
    '-b', 'pipeline',
    '-m', 'txt',
    '-l', 'ch',
    '--device', 'cpu',
]

print(f'Starting CLI parse...', flush=True)
t = time.time()

try:
    mineru_main(args=args, standalone_mode=False)
    print(f'CLI PARSE SUCCESS! dt={time.time()-t:.1f}s', flush=True)
    import glob
    md_files = glob.glob(os.path.join(output_dir, '**', '*.md'), recursive=True)
    print(f'Generated MD: {md_files}', flush=True)
    for f in md_files:
        with open(f, 'r', encoding='utf-8') as fp:
            content = fp.read()
            print(f'\n=== {f} ({len(content)} chars) ===')
            print(content[:1000])
except click.exceptions.Exit as e:
    print(f'Exit code: {e.exit_code}, dt={time.time()-t:.1f}s', flush=True)
except Exception as e:
    print(f'ERROR: {type(e).__name__}: {e}', flush=True)
    import traceback
    traceback.print_exc()