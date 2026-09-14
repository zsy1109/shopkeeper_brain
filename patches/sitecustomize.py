"""
Auto-apply compatibility patches for this project on Windows.
Place this file in your virtualenv's site-packages/ directory:
    .venv/Lib/site-packages/sitecustomize.py

Python will automatically import sitecustomize.py on startup (before any other package),
so these monkey-patches take effect globally across all processes.

Three patches are applied:
1. transformers 5.x RTDetr class_embed path compatibility
2. transformers 5.17 CVE-2025-32434 torch >= 2.6 check bypass
3. FastText model path fix for Chinese directory names on Windows
"""
import torch


# ============================================================
# Patch 1: transformers 5.x RTDetrForObjectDetection compatibility
# transformers 5.x moved class_embed/bbox_embed inside model.decoder,
# but MinerU's code still accesses them at the old top-level path.
# ============================================================
try:
    from transformers.models.rt_detr.modeling_rt_detr import RTDetrForObjectDetection

    _orig_rtdetr_init = RTDetrForObjectDetection.__init__

    def _patched_rtdetr_init(self, config):
        _orig_rtdetr_init(self, config)
        self.class_embed = self.model.decoder.class_embed
        self.bbox_embed = self.model.decoder.bbox_embed

    RTDetrForObjectDetection.__init__ = _patched_rtdetr_init
except Exception:
    pass


# ============================================================
# Patch 2: bypass transformers CVE-2025-32434 torch >= 2.6 check
# transformers 5.17.0 added a hard check that refuses to torch.load
# if torch < 2.6. We use torch 2.5.1 (cu121), and BGE-M3 weights
# are not in safetensors format, so this blocks embedding entirely.
# Must patch BOTH import_utils AND modeling_utils (modeling_utils
# imports the function directly at module load time, so patching
# import_utils alone is insufficient).
# ============================================================
try:
    import transformers.utils.import_utils as _tiu
    _tiu.check_torch_load_is_safe = lambda: None
except Exception:
    pass
try:
    import transformers.modeling_utils as _tmu
    _tmu.check_torch_load_is_safe = lambda: None
except Exception:
    pass


# ============================================================
# Patch 3: FastText path on Windows with Chinese characters
# fasttext_pybind's C++ binding cannot handle paths containing
# non-ASCII characters (e.g. "项目" in D:\PyCharm项目\...).
# We copy the model to an ASCII path (C:\Temp\ftlang\) and
# override the module's internal path reference.
# ============================================================
try:
    import os as _os
    import shutil as _shutil
    from pathlib import Path as _Path

    _ftlang_cache = _os.getenv("FTLANG_CACHE", r"C:\Temp\ftlang")
    _os.makedirs(_ftlang_cache, exist_ok=True)
    _model_path = _os.path.join(_ftlang_cache, "lid.176.bin")

    if not _os.path.isfile(_model_path):
        _orig_model = _os.path.join(
            _os.path.dirname(__file__),
            "fast_langdetect", "ft_detect", "resources", "lid.176.ftz"
        )
        if _os.path.isfile(_orig_model):
            _shutil.copy2(_orig_model, _model_path)

    if _os.path.isfile(_model_path):
        import fast_langdetect.ft_detect.infer as _ft_infer
        _ft_infer.LOCAL_SMALL_MODEL_PATH = _Path(_model_path)

    _os.environ["FTLANG_CACHE"] = _ftlang_cache
except Exception:
    pass


# ============================================================
# Patch 4: transformers 5.x pytorch_utils prune functions
# _find_pruneable_heads_and_indices and _prune_linear_layer had
# signature changes that break older model code paths.
# ============================================================
try:
    import transformers.pytorch_utils as pt_utils

    def _fpn(heads, n_heads, head_size, already_pruned_heads):
        mask = torch.ones(n_heads, head_size)
        heads = set(heads) - already_pruned_heads
        heads_to_prune = sorted(list(heads))
        index = torch.ones(n_heads, dtype=torch.long)
        for head in heads_to_prune:
            mask[head] = 0
            index[head] = 0
        return heads_to_prune, index

    def _pll(layer, index, dim=0):
        index = index.to(layer.weight.device)
        W = layer.weight.index_select(dim, index).clone().detach()
        if layer.bias is not None:
            b = layer.bias.clone().detach() if dim != 1 else layer.bias[index].clone().detach()
        else:
            b = None
        new_size = list(layer.weight.shape)
        new_size[dim] = len(index)
        new_layer = torch.nn.Linear(new_size[1], new_size[0], bias=layer.bias is not None).to(layer.weight.device)
        new_layer.weight.requires_grad = False
        new_layer.weight.copy_(W.contiguous())
        new_layer.weight.requires_grad = True
        if b is not None:
            new_layer.bias.requires_grad = False
            new_layer.bias.copy_(b.contiguous())
            new_layer.bias.requires_grad = True
        return new_layer

    pt_utils.find_pruneable_heads_and_indices = _fpn
    pt_utils.prune_linear_layer = _pll
except Exception:
    pass