# 数据集与数据加载器
import numpy as np
from torch.utils.data import Dataset,DataLoader,Subset
from torchvision import transforms
from PIL import Image
from torch.config import IMAGE_SIZE,BATCH_SIZE,NUM_WORKERS
def get_train_transforms():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE+32,IMAGE_SIZE+32)),
        transforms.RandomCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2,contrast=0.2,saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]),
    ])
def get_val_transforms():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]),
    ])
class CustomImageDataset(Dataset):
    def __init__(self,image_paths,labels,transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self,idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image,label
def create_dataloaders(train_indices,val_indices,full_dataset):
    train_subset = Subset(full_dataset,train_indices)
    val_subset = Subset(full_dataset,val_indices)
    val_subset.dataset.transform = get_val_transforms()
    train_loader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    return train_loader,val_loader
def load_dataset_from_dir(data_dir,transform=None):
    from torchvision.datasets import ImageFolder
    dataset = ImageFolder(root=str(data_dir),transform=transform)
    image_paths = [item[0] for item in dataset.samples]
    labels = [item[1] for item in dataset.samples]
    class_names = dataset.classes
    return image_paths,labels,class_names