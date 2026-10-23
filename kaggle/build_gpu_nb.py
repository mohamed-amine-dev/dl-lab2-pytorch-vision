import json
import pathlib

import nbformat

out = pathlib.Path("kaggle/gpu_nb")
out.mkdir(parents=True, exist_ok=True)
code = (
    "import torch, json\n"
    'info = {"torch": torch.__version__, "cuda": torch.cuda.is_available(), '
    '"name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}\n'
    "print(info)\n"
    'json.dump(info, open("/kaggle/working/gpu_info.json", "w"))\n'
)
nb = nbformat.v4.new_notebook()
nb.cells = [nbformat.v4.new_code_cell(code)]
nbformat.write(nb, out / "gpu_nb.ipynb")
meta = {
    "id": "ouallalimohamedamine/gpu-check-notebook-dl-lab2",
    "title": "gpu-check-notebook-dl-lab2",
    "code_file": "gpu_nb.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "enable_internet": True,
    "dataset_sources": [],
    "competition_sources": [],
    "kernel_sources": [],
}
(out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print("built", out)
