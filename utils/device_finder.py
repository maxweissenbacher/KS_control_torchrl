import torch


def network_device(cfg):
    if cfg.network.auto_detect_device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        return torch.device(str(cfg.network.device))

