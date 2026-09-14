# MinerU ONNX Runtime Patches

MinerU 3.4.5 自带 ONNX Runtime 1.30.0，要求 CUDA 13.x + cuDNN 9.x。多数开发环境用 CUDA 12.x，
会报 `Require cuDNN 9.* and CUDA 13.*` 或 `CUDA error: out of memory`。

此外，Windows 上多进程重复加载 cublas64_12.dll 会耗尽虚拟内存。

修复方案：强制所有 MinerU ONNX 模型使用 CPUExecutionProvider。

---

## Patch 1: `mineru/model/table/rec/onnxruntime_provider.py`

**改动**：把 `build_table_onnx_providers()` 改成只返回 CPU provider。

**原代码**（约第 44 行）：
```python
def build_table_onnx_providers(
    available_providers: Sequence[str],
) -> List[Tuple[str, dict[str, Any]]]:
    cpu_provider = _build_cpu_provider()
    if _normalize_device(get_device()) == "cuda" and CUDA_PROVIDER in available_providers:
        return [_build_cuda_provider(), cpu_provider]
    return [cpu_provider]
```

**改后**：
```python
def build_table_onnx_providers(
    available_providers: Sequence[str],
) -> List[Tuple[str, dict[str, Any]]]:
    """根据 MinerU 当前设备为表格 ONNX 模型选择 onnxruntime providers。"""
    cpu_provider = _build_cpu_provider()
    # ONNX Runtime 1.30.0 requires CUDA 13.x + cuDNN 9.x.
    # Always use CPU to avoid CUDA OOM on incompatible setups.
    return [cpu_provider]
```

---

## Patch 2: `mineru/model/table/cls/paddle_table_cls.py`

**改动**：`PaddleTableClsModel.__init__` 的 InferenceSession 显式指定 CPU provider。

**原代码**（约第 18 行）：
```python
self.sess = onnxruntime.InferenceSession(
    os.path.join(...), ModelPath.paddle_table_cls),
)
```

**改后**：
```python
self.sess = onnxruntime.InferenceSession(
    os.path.join(...), ModelPath.paddle_table_cls),
    providers=['CPUExecutionProvider'],
)
```

---

## 安装方式

修改你的虚拟环境中 mineru 对应的两个文件，路径：

```
.venv/Lib/site-packages/mineru/model/table/rec/onnxruntime_provider.py
.venv/Lib/site-packages/mineru/model/table/cls/paddle_table_cls.py
```

这两个改动是 MinerU 源码级别的，每次重装 MinerU 后都需要重新应用。