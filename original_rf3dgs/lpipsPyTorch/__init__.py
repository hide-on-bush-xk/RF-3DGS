import torch

from .modules.lpips import LPIPS


def lpips(x: torch.Tensor,
          y: torch.Tensor,
          net_type: str = 'alex',
          version: str = '0.1'):
    r"""Function that measures
    Learned Perceptual Image Patch Similarity (LPIPS).

    Arguments:
        x, y (torch.Tensor): the input tensors to compare.
        net_type (str): the network type to compare the features:
                        'alex' | 'squeeze' | 'vgg'. Default: 'alex'.
        version (str): the version of LPIPS. Default: 0.1.

    Lower is better (0 means perceptually identical). metrics.py calls this with
    net_type='vgg', which is the variant reported in the 3DGS paper.

    Note: the backbone is rebuilt and moved to the GPU on *every* call, and the
    pretrained weights are fetched through torch.hub the first time (cached in
    ~/.cache/torch/hub/checkpoints afterwards). Evaluating a whole test set one
    image at a time therefore pays that construction cost per image; hoisting the
    LPIPS object out of the loop is the obvious speed-up if this ever matters.
    """
    device = x.device
    # LPIPS = feature distance between the two images under a frozen pretrained
    # backbone, with per-channel weights learned to match human judgements.
    criterion = LPIPS(net_type, version).to(device)
    return criterion(x, y)
