"""Test MinerU via subprocess isolation."""
import subprocess, sys, os, time

env = os.environ.copy()
env.setdefault('MINERU_DEVICE_MODE', 'cpu')
env.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
env.setdefault('OPENBLAS_NUM_THREADS', '1')
env.setdefault('OMP_NUM_THREADS', '1')
env.setdefault('MKL_NUM_THREADS', '1')

cmd = [
    sys.executable,
    '-m', 'mineru.cli.client',
    '-p', r'D:\PyCharm项目\shopkeeper_brain_1\docs\万用表的使用_origin.pdf',
    '-o', r'D:\PyCharm项目\shopkeeper_brain_1\data\tmp\mineru_subprocess_test',
    '-b', 'pipeline',
    '-m', 'txt',
    '-l', 'ch',
]

print('Starting subprocess test...')
print(f'CMD: {" ".join(cmd)}')
t0 = time.time()

result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=300)
print(f'Exit code: {result.returncode}')
print(f'Time: {time.time()-t0:.1f}s')

stdout = result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout
stderr = result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr

print('\n--- STDOUT (last 2000 chars) ---')
print(stdout)
print('\n--- STDERR (last 2000 chars) ---')
print(stderr)

# Check output
out_dir = r'D:\PyCharm项目\shopkeeper_brain_1\data\tmp\mineru_subprocess_test'
if os.path.isdir(out_dir):
    print('\n--- Output files ---')
    for root, dirs, files in os.walk(out_dir):
        for f in files:
            print(f'  {os.path.join(root, f)}')