import torch
import sys

def check_cuda_availability():
    print("Python version:", sys.version)
    print("PyTorch version:", torch.__version__)
    
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        print("CUDA is available. GPU can be used for computations.")
        print(f"Number of GPUs available: {torch.cuda.device_count()}")
        print(f"Current GPU device: {torch.cuda.current_device()}")
        print(f"GPU name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        print(f"CUDA version reported by PyTorch: {torch.version.cuda}")
    else:
        print("CUDA is not available. Computations will use CPU.")
        print(f"CUDA version reported by PyTorch: {torch.version.cuda}")
        print("Possible reasons for CUDA not being detected:")
        print("- PyTorch might not be installed with CUDA support for the correct version.")
        print("- NVIDIA drivers might be outdated or incompatible with the installed CUDA toolkit.")
        print("- There could be a mismatch between the CUDA toolkit version (12.9) and PyTorch's supported versions.")

if __name__ == "__main__":
    check_cuda_availability()
