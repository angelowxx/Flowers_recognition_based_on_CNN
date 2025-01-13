import torch
from torchvision import transforms


class ToFrequencySpectrumAndCrop:
    def __init__(self, crop_size=128):
        """
        Initialize the transform with a crop size.

        Args:
            crop_size (int): Size of the central crop (crop_size x crop_size).
        """
        self.crop_size = crop_size

    def __call__(self, x):
        """
        Apply FFT, shift the frequency spectrum, and crop the central region.

        Args:
            x (Tensor): Input image tensor (C x H x W).

        Returns:
            Tensor: Cropped frequency spectrum (C x crop_size x crop_size).
        """
        # Ensure input is a tensor
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input should be a PyTorch tensor")

        # Apply FFT on height and width dimensions
        freq = torch.fft.fft2(x, dim=(-2, -1))

        # Shift zero frequency to the center
        shifted = torch.fft.fftshift(freq, dim=(-2, -1))

        # Compute the magnitude of the frequency spectrum
        magnitude = torch.abs(shifted)

        # Crop the central region
        h, w = magnitude.shape[-2:]
        crop_h, crop_w = self.crop_size, self.crop_size

        start_h = (h - crop_h) // 2
        start_w = (w - crop_w) // 2
        cropped = magnitude[..., start_h:start_h + crop_h, start_w:start_w + crop_w]

        return cropped


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
    transforms.RandomAffine(degrees=(-90, 90), translate=(0.2, 0.2), shear=20, scale=(0.8, 1.2)),  #
    transforms.Resize(size=(128, 128)),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.05),
])



