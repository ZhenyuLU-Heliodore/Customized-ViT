from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


def create_optimizer(
        params,
        lr=1e-3,
        optim_type="AdamW",
):
    # TODO: Add more types
    if optim_type == "AdamW":
        optimizer = AdamW(params, lr)
    else:
        raise ValueError("Got wrong optimizer type")

    return optimizer


def create_lr_scheduler(
        optimizer,
        scheduler_type="cosine",
        T_max=None,
        eta_min=1e-7,
):
    # TODO: Add more types
    if scheduler_type == "cosine":
        scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=T_max, eta_min=eta_min)
    else:
        raise ValueError("Got wrong scheduler type")

    return scheduler