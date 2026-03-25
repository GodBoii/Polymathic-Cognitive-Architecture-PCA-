import json

import torch


def main() -> None:
    payload = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "device_count": torch.cuda.device_count(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
