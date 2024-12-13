from torchvision import transforms
import random

resize_to_64x64 = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

resize_and_colour_jitter = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor()
])


translation_rotation = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomAffine(degrees=90, translate=(0.2, 0.2), shear=20, scale=(0.8, 1.2)),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
])

cropping_img = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomCrop(size=(64, 64)),
    transforms.ToTensor()
])

data_augmentation_pipline = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomAffine(degrees=(0, 90), translate=(0.1, 0.2), shear=10, scale=(0.9, 1.1)),  #
    transforms.RandomCrop(size=(64, 64)),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor()
])



