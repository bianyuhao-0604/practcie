# train.py
import argparse, math, os
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler

from augmentations import simclr_augment
from model import SimCLR
from nt_xent import NTXentLoss, contrastive_accuracy


# ---------- LARS：大批量(≥512)建议启用；小批量可用 AdamW ----------
class LARS(torch.optim.Optimizer):
    """Layer-wise Adaptive Rate Scaling (You et al., 2017)。"""
    def __init__(self, params, lr, momentum=0.9, weight_decay=1e-6,
                 trust_coefficient=1e-3, eps=1e-8):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                        trust_coefficient=trust_coefficient, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            wd, tc, eps = group["weight_decay"], group["trust_coefficient"], group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad
                p_norm = p.norm()
                g_norm = d_p.norm()
                # LARS 核心逐层缩放：trust = ||w|| / (||w|| + wd*||w|| + ||g|| + eps)
                if p_norm != 0 and g_norm != 0:
                    local_lr = tc * p_norm / (p_norm + wd * p_norm + g_norm + eps)
                    d_p = d_p.add(p, alpha=wd)
                    d_p = d_p * local_lr
                buf = self.state[p].get("momentum_buffer")
                if buf is None:
                    buf = torch.zeros_like(p)
                    self.state[p]["momentum_buffer"] = buf
                buf.mul_(group["momentum"]).add_(d_p)
                p.add_(buf, alpha=-group["lr"])


def exclude_bn_bias(model):
    """weight decay 不加在 BN 参数与 bias 上（论文惯例）。"""
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 1 or "bn" in n or ".downsample.1" in n or n.endswith(".bias"):
            no_decay.append(p)
        else:
            decay.append(p)
    return [{"params": decay, "weight_decay": 1e-6},
            {"params": no_decay, "weight_decay": 0.0}]


def get_lr(step, total_steps, base_lr, warmup_steps):
    """线性 warmup + 余弦衰减（论文默认 10 epochs warmup）。"""
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))


def make_loader(args, train=True, shuffle=True, ddp=False):
    tf = simclr_augment(size=32, s=args.color_s, blur=args.blur)
    ds = datasets.CIFAR10(args.data, train=train, download=True, transform=tf)
    sampler = DistributedSampler(ds, shuffle=shuffle) if ddp else None
    return DataLoader(ds, batch_size=args.batch_size,
                      shuffle=(sampler is None and shuffle),
                      sampler=sampler, num_workers=args.workers,
                      pin_memory=True, drop_last=True, persistent_workers=True), sampler


def train(args):
    ddp = int(os.environ.get("WORLD_SIZE", 1)) > 1
    rank = int(os.environ.get("RANK", 0))
    device = f"cuda:{rank % torch.cuda.device_count()}"
    if ddp:
        dist.init_process_group("nccl")
        torch.cuda.set_device(device)

    torch.manual_seed(args.seed + rank)
    writer = SummaryWriter(f"runs/pretrain") if rank == 0 else None

    model = SimCLR(args.arch, args.dataset).to(device)
    if ddp:
        model = DDP(model, device_ids=[rank % torch.cuda.device_count()])
    raw_model = model.module if ddp else model

    loader, sampler = make_loader(args, ddp=ddp)

    params = exclude_bn_bias(raw_model)
    if args.batch_size >= 512:
        base_lr = 0.075 * math.sqrt(args.batch_size)     # sqrt LR scaling（附录 B.1）
        opt = LARS(params, lr=base_lr, momentum=0.9)
    else:
        base_lr = 1e-3 * math.sqrt(args.batch_size / 256)
        opt = torch.optim.AdamW(params, lr=base_lr)

    criterion = NTXentLoss(temperature=args.temperature).to(device)
    scaler = GradScaler(device) if args.amp else None

    steps_per_epoch = len(loader)
    total_steps = steps_per_epoch * args.epochs
    warmup = steps_per_epoch * args.warmup_epochs
    step = 0

    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        ep_loss, ep_acc = 0.0, 0.0
        for (x1, x2), _ in loader:
            x1, x2 = x1.to(device, non_blocking=True), x2.to(device, non_blocking=True)
            for g in opt.param_groups:
                g["lr"] = get_lr(step, total_steps, base_lr, warmup)
            opt.zero_grad(set_to_none=True)

            amp_ctx = autocast(device, enabled=bool(scaler))
            with amp_ctx:
                z1, _ = model(x1, return_h=True)
                z2, _ = model(x2, return_h=True)
                loss = criterion(z1, z2)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 10.)
                scaler.step(opt); scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 10.)
                opt.step()

            acc = contrastive_accuracy(z1.float(), z2.float())
            ep_loss += loss.item(); ep_acc += acc
            if rank == 0 and step % args.log_every == 0:
                writer.add_scalar("pretrain/loss", loss.item(), step)
                writer.add_scalar("pretrain/contrastive_acc", acc, step)
                writer.add_scalar("pretrain/lr", opt.param_groups[0]["lr"], step)
            step += 1

        if rank == 0:
            n = steps_per_epoch
            print(f"[epoch {epoch:3d}] loss={ep_loss/n:.4f} "
                  f"contrastive_acc={ep_acc/n:.4f}")
            writer.add_scalar("pretrain/epoch_loss", ep_loss / n, epoch)
            torch.save({"model": raw_model.state_dict(),
                        "epoch": epoch, "args": vars(args)},
                       f"runs/simclr_{args.dataset}_last.pth")
        if ddp:
            dist.barrier()

    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="./datasets")
    p.add_argument("--dataset", default="cifar10")
    p.add_argument("--arch", default="resnet18")
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.5)   # CIFAR-10 用 0.5
    p.add_argument("--color_s", type=float, default=0.5)
    p.add_argument("--blur", action="store_true")
    p.add_argument("--warmup_epochs", type=int, default=10)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", action="store_true")
    train(p.parse_args())
