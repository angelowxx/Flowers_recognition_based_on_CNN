from torchvision import transforms
import random

resize_to_64x64 = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

resize_and_colour_jitter = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor()
])

data_augmentation_pipline = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomCrop(size=(64, 64)),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
    transforms.RandomAffine(degrees=(-90, 90), translate=(0.2, 0.2), shear=20, scale=(0.8, 1.2)),  #
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor()
])

# Transformation that includes translation
translation_rotation = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomAffine(degrees=180, translate=(0.2, 0.2)),  # Translate up to 20% horizontally/vertically
    transforms.ToTensor(),
])


