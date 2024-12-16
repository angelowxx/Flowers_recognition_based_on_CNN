import torch
from torchvision import transforms
import random

resize_to_64x64 = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

to_tensor = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.05)
])

add_noise = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.05)
])

resize_and_colour_jitter = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ColorJitter(brightness=0.5, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.ToTensor(),
])


translation_rotation = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomAffine(degrees=180, translate=(0.3, 0.3), shear=25, scale=(0.7, 1.3)),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor()
])

vertical_flipping = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomVerticalFlip(p=1),
    transforms.ToTensor()
])

horizontal_flipping = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(p=1),
    transforms.ToTensor()
])

cropping_img = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.RandomCrop(size=(64, 64)),
    transforms.ToTensor()
])

data_augmentation_pipline = transforms.Compose([
    transforms.RandomCrop(size=(64, 64)),
    transforms.RandomAffine(degrees=(0, 90), translate=(0.3, 0.3), shear=30, scale=(0.7, 1.3)),  #
    transforms.ColorJitter(brightness=0.5, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
])



