from torchvision import transforms

resize_to_64x64 = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

resize_and_colour_jitter = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor()
])

# Transformation that includes translation
translation_rotation = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),  # Translate up to 20% horizontally/vertically
    transforms.ToTensor(),
])


