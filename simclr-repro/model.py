# model.py
import torch.nn as nn
import torchvision


class ProjectionHead(nn.Module):
    """2 层 MLP 投影头：h(2048/512) → hidden → 128。训练完成后丢弃。"""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )
    def forward(self, x):
        return self.net(x)


class SimCLR(nn.Module):
    def __init__(self, arch: str = "resnet18", dataset: str = "cifar10",
                 projection_dim: int = 128):
        super().__init__()
        if dataset == "cifar10":
            # CIFAR 版：3x3 stem + 去 maxpool（论文附录 B.9），fc 输出当 h（512 维）
            backbone = torchvision.models.resnet18(weights=None, num_classes=512)
            backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1,
                                       padding=1, bias=False)
            backbone.maxpool = nn.Identity()
            feat_dim = 512
        else:
            # ImageNet 版：标准 ResNet-50，avgpool 后 h 为 2048 维
            backbone = torchvision.models.resnet50(weights=None, num_classes=2048)
            feat_dim = 2048
        self.encoder = backbone                    # f(·)，下游任务用它输出 h
        hidden = 512 if dataset == "cifar10" else 2048
        self.projector = ProjectionHead(feat_dim, hidden, projection_dim)  # g(·)

    def forward(self, x, return_h: bool = False):
        h = self.encoder(x)                        # 训练完做下游时用这个
        z = self.projector(h)                      # 损失只定义在 z 上
        return (z, h) if return_h else z
