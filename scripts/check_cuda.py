"""
Check PyTorch and CUDA setup. Run: python scripts/check_cuda.py
"""
import sys
print("Python:", sys.executable)
print("Python version:", sys.version)

try:
    import torch
    print("PyTorch version:", torch.__version__)
    print("torch.cuda.is_available():", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("CUDA version (runtime):", torch.version.cuda)
    else:
        print("\n--- PyTorch is using CPU (no CUDA). To enable GPU on Windows: ---")
        print("1. Uninstall current torch first:")
        print("   pip uninstall torch torchvision torchaudio")
        print("2. Install PyTorch with CUDA (pick ONE that matches your NVIDIA driver):")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        print("   # or for older drivers: .../whl/cu118")
        print("3. Check your driver: nvidia-smi (in terminal)")
        print("   CUDA 12.x driver -> use cu121 ; CUDA 11.x -> use cu118")
        print("4. Restart the terminal / IDE after installing, then run this script again.")
except Exception as e:
    print("Error importing torch:", e)
