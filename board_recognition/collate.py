import torch

def batch_collate_fn(batch):
    """
    Minimal collate: just stack images [B,C,H,W] on CPU, and keep a list of targets.
    """
    images, targets = list(zip(*batch))
    batch_imgs = torch.stack(images, dim=0)  # CPU tensor
    return batch_imgs, list(targets)
