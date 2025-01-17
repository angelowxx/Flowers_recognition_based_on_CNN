import random

import torch
from torchvision import transforms
from torchvision.transforms import functional as F
from PIL import Image


class RepeatImageTransform:
    def __init__(self, min_repeat=2, max_repeat=3):
        """
        Initialize the RepeatImageTransform.
        Args:
            min_repeat (int): Minimum number of times to repeat along width and height.
            max_repeat (int): Maximum number of times to repeat along width and height.
        """
        self.min_repeat = min_repeat
        self.max_repeat = max_repeat

    def __call__(self, img):
        """
        Repeat the image randomly along width and height.
        Args:
            img (PIL.Image): Input image.
        Returns:
            PIL.Image: Repeated image.
        """
        # Randomly decide the repeat factors for height and width
        repeat_h = random.randint(self.min_repeat, self.max_repeat)
        repeat_w = random.randint(self.min_repeat, self.max_repeat)

        # Get original dimensions of the image
        width, height = img.size

        # Create a new blank image to hold the repeated pattern
        repeated_img = Image.new(img.mode, (width * repeat_w, height * repeat_h))

        # Paste the image repeatedly
        for i in range(repeat_h):
            for j in range(repeat_w):
                repeated_img.paste(img, (j * width, i * height))

        return repeated_img


# normalization parameters from imageNet
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

resize_to_64x64 = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

resize_to_128x128 = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.05),
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
    transforms.Resize((256, 256)),
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
    transforms.RandomAffine(degrees=(-90, 90), translate=(0.3, 0.3), shear=30, scale=(0.7, 1.3)),  #
    transforms.Resize(size=(128, 128)),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.05),
])

data_augmentation_pipline_repeated = transforms.Compose([
    transforms.RandomAffine(degrees=(-90, 90), translate=(0.3, 0.3), shear=30, scale=(0.7, 1.3)),  #
    RepeatImageTransform(),
    transforms.Resize(size=(128, 128)),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.05),
])



