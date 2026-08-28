#模型定义（预训练 + 微调）
import torch
import torch.nn as nn
from torchvision import models
from torch.config import MODEL_NAME,NUM_CLASSES,PRETRAINED,FREEZE_BACKBONE
def build_model():
    if MODEL_NAME == "resnet50":
        weights = models.Resnet50_Weights.DEFAULT if PRETRAINED else None
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features,NUM_CLASSES),
        )
    elif MODEL_NAME == 'resnet18':
        weights = models.ResNet18_Weights.DEFAULT if PRETRAINED else None
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features,NUM_CLASSES),
        )
    elif MODEL_NAME == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if PRETRAINED else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_fetures
        model.classifier[1] = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features,NUM_CLASSES),
        )
    else:
        raise ValueError(f"不支持的模型: {MODEL_NAME}")
    if FREEZE_BACKBONE:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True
    return model
def get_optimizer(model,lr,weight_decay):
    backbone_params = []
    head_params = []
    for name,param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "fc" in name or "classifier" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)
    optimizer = torch.optim.AdamW([
        {"params":backbone_params,"lr":lr*0.1},
        {"params":head_params,"lr":lr},
        ],weight_decay = weight_decay)
    return optimizer
def get_scheduler(optimizer,total_epochs):
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,T_max=total_epochs,eta_min=1e-7
    )