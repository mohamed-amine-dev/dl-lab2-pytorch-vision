import torch, platform, json

info = {
    "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "device_count": torch.cuda.device_count(),
    "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    "cuda_version": torch.version.cuda,
    "python": platform.python_version(),
}
print(json.dumps(info, indent=2))
with open("/kaggle/working/gpu_info.json", "w") as f:
    json.dump(info, f, indent=2)
