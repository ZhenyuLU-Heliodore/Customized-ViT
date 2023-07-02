import torch
import yaml
import os
import argparse

from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, SequentialSampler
from vit_classifier.modeling import VitMetaEncoder
from vit_classifier.train_utils import(
    init_process_group,
    distribute_model,
    create_optimizer,
    create_lr_scheduler,
    VitTrainLoopDDP
)
from vit_classifier.data_utils import (
    get_cifar_100_testset,
    get_cifar100_trainset,
)


def main(node_rank, local_rank, args):
    # set the environment
    init_process_group()
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda:{:d}'.format(local_rank))
    with open(args.kwargs_path, "r") as file:
        kwargs = yaml.load(file, Loader=yaml.FullLoader)

    # dataset and dataloader
    train_dataset = get_cifar100_trainset(args.data_dir)
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    valid_dataset = get_cifar_100_testset(args.data_dir)
    valid_sampler = SequentialSampler(valid_dataset)

    cfn = CollateFn().collate_func
    train_loader = DataLoader(
        train_dataset, batch_size=args.train_batch_size, sampler=train_sampler, num_workers=8, collate_fn=cfn,
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=args.valid_batch_size, sampler=valid_sampler, num_workers=16, collate_fn=cfn,
    )

    # initialize the model, optimizer and scheduler
    model = VitMetaEncoder(**kwargs["VitMetaEncoder"])
    if args.load_ckpt_path is None: # resume training
        ddp_model = distribute_model(model.to(device), device=device)
        optimizer = create_optimizer(list(ddp_model.parameters()), **kwargs["optimizer"])
        scheduler = create_lr_scheduler(optimizer, T_max=args.epochs*len(train_loader), **kwargs["lr_scheduler"])
    else:
        ckpt = torch.load(args.load_ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        ddp_model = distribute_model(model.to(device), device=device)
        optimizer = create_optimizer(list(ddp_model.parameters()), **kwargs["optimizer"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler = create_lr_scheduler(optimizer, T_max=args.epochs*len(train_loader), **kwargs["lr_scheduler"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    train_loop = VitTrainLoopDDP(
        model=ddp_model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        valid_loader=valid_loader,
        local_rank=local_rank,
        device=device,
        tb_log_dir=args.tb_log_dir,
    )

    if node_rank == 0 and local_rank == 0 and args.start_epoch == 0:
        train_loop.save_init(args.ckpt_dir)
    for epoch in range(args.start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        train_loop.train(epoch)
        if node_rank == 0 and local_rank == 0:
            train_loop.validate(epoch)
        if node_rank == 0 and local_rank == 0 and (epoch + 1) % args.saving_interval == 0:
            train_loop.save_ckpt(epoch, args.ckpt_dir)


class DemoDataset(torch.utils.data.Dataset):
    # train on few batches to test potential bugs
    def __init__(self, cifar100):
        self.dataset = cifar100
    def __getitem__(self, idx):
        nid = idx + 5000
        return self.dataset[nid]
    def __len__(self):
        return 64


class CollateFn:
    def __init__(self):
        # sometimes need to input some init parameters.
        pass

    def collate_func(self, batch_data):
        image_list, label_list = [], []
        for data in batch_data:
            image_list.append(data[0])
            label_list.append(data[1])

        return (
            torch.stack(image_list, dim=0),
            torch.as_tensor(label_list, dtype=torch.long)
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model args
    parser.add_argument("--kwargs_path", default="/home/ubuntu/projects/vit-for-cifar100/configs/default.yml", type=str)
    parser.add_argument("--load_ckpt_path", default=None, type=str)

    # dataset args
    parser.add_argument("--data_dir", default="/home/ubuntu/data/cifar100", type=str)

    # saving args
    parser.add_argument("--tb_log_dir", default="/home/ubuntu/logs/vit/temp", type=str)
    parser.add_argument("--ckpt_dir", default="/home/ubuntu/ckpt/vit/temp", type=str)
    parser.add_argument("--saving_interval", default=1, type=int)

    # training args
    parser.add_argument("--train_batch_size", default=64, type=int)
    parser.add_argument("--valid_batch_size", default=126, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--start_epoch", default=0, type=int)


    args = parser.parse_args()
    if os.environ.get("GROUP_RANK"):
        torch.multiprocessing.set_start_method("spawn")
        main(
            int(os.environ["GROUP_RANK"]),
            int(os.environ["LOCAL_RANK"]),
            args,
        )
    else:
        raise RuntimeError