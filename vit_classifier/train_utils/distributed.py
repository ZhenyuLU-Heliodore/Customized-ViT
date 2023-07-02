from torch import distributed
from torch.nn.parallel import DistributedDataParallel


def init_process_group():
    distributed.init_process_group(
        backend='nccl',
        init_method='env://'
    )

def distribute_model(model, device, find_unused_paras=True):
    if device.type == 'cpu':
        model = DistributedDataParallel(model)
    else:
        model = DistributedDataParallel(
            model, device_ids=[device], output_device=device, find_unused_parameters=find_unused_paras,
        )

    return model