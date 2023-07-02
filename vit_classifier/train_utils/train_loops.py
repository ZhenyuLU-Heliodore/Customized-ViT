import torch
import os

from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from ..modeling import compute_loss


class VitTrainLoopDDP:
    def __init__(
            self,
            model: DistributedDataParallel,
            optimizer: Optimizer,
            local_rank: int,
            device: torch.device,
            train_loader: DataLoader,
            tb_log_dir: str,
            scheduler: LRScheduler = None,
            valid_loader: DataLoader = None,
    ):
        self.model = model
        self.local_rank = local_rank
        self.device = device
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.optimizer = optimizer
        self.tb_log_dir = tb_log_dir
        self.scheduler = scheduler
        self.batch_train_losses = 0

        if self.local_rank == 0:
            os.makedirs(tb_log_dir, exist_ok=True)
            self.writer = SummaryWriter(tb_log_dir)

    def train(self, epoch):
        # train an epoch
        self.model.train()
        self.batch_train_losses = 0

        for batch_id, data in enumerate(self.train_loader):
            images, labels = data[0].to(self.device), data[1].to(self.device)
            global_batch_id = batch_id + epoch * len(self.train_loader)

            out = self.model(images)
            self.optimizer.zero_grad()
            loss = compute_loss(out, labels)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(list(self.model.parameters()), 1.0)
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            self.batch_train_losses += loss.item()
            if batch_id % 10 == 0:
                print(
                    'Rank: %d, epoch: %d, batch: %d / %d, loss: %.6f' % (
                        self.local_rank, epoch, batch_id, len(self.train_loader), loss
                    )
                )
            if self.local_rank == 0:
                self.writer.add_scalar('Train_loss', loss.item(), global_batch_id)

            torch.cuda.empty_cache()

    def validate(self, epoch):
        # validate an epoch
        assert self.valid_loader is not None, "Got empty validation dataloader!"
        valid_losses, accs, num_samples = 0, 0, 0
        self.model.eval()

        for batch_id, data in enumerate(self.valid_loader):
            images, labels = data[0].to(self.device), data[1].to(self.device)

            with torch.no_grad():
                out = self.model(images)
                loss = compute_loss(out, labels)

            valid_losses += loss.item()
            accs += (out.argmax(dim=-1) == labels).to(torch.float).sum()
            num_samples += images.shape[0]

        loss, acc = valid_losses / len(self.valid_loader), accs / num_samples
        if self.local_rank == 0:
            self.writer.add_scalar('Valid_loss', loss, epoch)
            self.writer.add_scalar('Accuracy', acc, epoch)
        print(
            'Rank: %d, epoch: %d, loss: %.6f, acc: %.2f' % (
                self.local_rank, epoch, loss, acc
            )
        )
        torch.cuda.empty_cache()

    def save_init(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        init_dict = {
            "model_init": self.model.module,
            "optimizer_init": self.optimizer,
        }
        if self.scheduler is not None:
            init_dict["scheduler_init"] = self.scheduler

        torch.save(init_dict, os.path.join(save_dir, "init.pth"))

    def save_ckpt(self, epoch, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        ckpt = {
            "model_state_dict": self.model.module.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss": self.batch_train_losses / len(self.train_loader),
        }
        if self.scheduler is not None:
            ckpt["scheduler_state_dict"] = self.scheduler.state_dict()

        torch.save(ckpt, os.path.join(save_dir, "ckpt_epoch_{}.pth".format(epoch)))