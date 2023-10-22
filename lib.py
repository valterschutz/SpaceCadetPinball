import torch

def get_device():
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        if num_gpus >= 2:
            return "cuda:1"  # Return "cuda:1" if 2 or more GPUs are available
        else:
            return "cuda"  # Return "cuda" if only 1 GPU is available
    else:
        return "cpu"  # Return "cpu" if no GPUs are available

device = get_device()
