import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import torch.nn.functional as F


class OneHotImageFolder(Dataset):
    def __init__(self, root, transform=None, num_classes=None):
        self.dataset = ImageFolder(root=root, transform=transform)
        self.num_classes = num_classes if num_classes else len(self.dataset.classes)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        # Convert label to one-hot
        one_hot_label = F.one_hot(torch.tensor(label), num_classes=self.num_classes).float()
        return img, one_hot_label