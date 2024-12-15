from torchvision import transforms
import random

resize_to_64x64 = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

resize_and_colour_jitter = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
    transforms.ToTensor()
])


translation_rotation = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomAffine(degrees=90, translate=(0.3, 0.3), shear=25, scale=(0.7, 1.3)),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
])

cropping_img = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.RandomCrop(size=(64, 64)),
    transforms.ToTensor()
])

data_augmentation_pipline = transforms.Compose([
    transforms.RandomCrop(size=(64, 64)),
    transforms.RandomAffine(degrees=(0, 90), translate=(0.3, 0.3), shear=30, scale=(0.7, 1.3)),  #
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor()
])



