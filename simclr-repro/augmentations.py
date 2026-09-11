# augmentations.py
import torch
import torchvision.transforms as T

CIFAR_MEAN, CIFAR_STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


class TwoCropTransform:
    """把同一个增强算子独立调用两次，产生正样本对的两个视图。"""
    def __init__(self, transform):
        self.transform = transform
    def __call__(self, x):
        return self.transform(x), self.transform(x)


def simclr_augment(size: int = 32, s: float = 0.5,
                   blur: bool = False,
                   mean=CIFAR_MEAN, std=CIFAR_STD) -> TwoCropTransform:
    """
    s: 颜色抖动强度。CIFAR-10 用 0.5，ImageNet 用 1.0（论文 Table 1）。
    blur: CIFAR-10 建议关；ImageNet 建议开（论文附录 B.9）。
    """
    color_jitter = T.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    ops = [
        T.RandomResizedCrop(size=size, scale=(0.08, 1.0),
                            ratio=(3 / 4, 4 / 3)),          # 论文附录 A 的裁剪参数
        T.RandomHorizontalFlip(p=0.5),
        T.RandomApply([color_jitter], p=0.8),
        T.RandomGrayscale(p=0.2),
    ]
    if blur:
        # 核尺寸约为图像边长的 10%（CIFAR 32→3，ImageNet 224→23）
        ops.append(T.RandomApply(
            [T.GaussianBlur(kernel_size=max(3, size // 10 | 1), sigma=(0.1, 2.0))],
            p=0.5))
    ops += [T.ToTensor(), T.Normalize(mean, std)]
    return TwoCropTransform(T.Compose(ops))
