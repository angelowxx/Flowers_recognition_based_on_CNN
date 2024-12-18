import torch
from torchvision import transforms

# normalization parameters from imageNet
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

resize_to_64x64 = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    normalize
])

resize_to_128x128 = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

to_tensor = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.05)
])

add_noise = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.05)
])

resize_and_colour_jitter = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ColorJitter(brightness=0.5, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.ToTensor(),
])


translation_rotation = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomAffine(degrees=90, translate=(0.3, 0.3), shear=30, scale=(0.7, 1.3)),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
])

vertical_flipping = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomVerticalFlip(p=1),
    transforms.ToTensor()
])

horizontal_flipping = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(p=1),
    transforms.ToTensor()
])

cropping_img = transforms.Compose([
    transforms.Resize((192, 192)),
    transforms.RandomCrop(size=(128, 128)),
    transforms.ToTensor()
])

data_augmentation_pipline = transforms.Compose([
    transforms.RandomCrop(size=(192, 192)),
    transforms.RandomAffine(degrees=(0, 90), translate=(0.3, 0.3), shear=30, scale=(0.7, 1.3)),  #
    transforms.ColorJitter(brightness=0.5, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomCrop(size=(128, 128)),
    transforms.ToTensor(),
])



