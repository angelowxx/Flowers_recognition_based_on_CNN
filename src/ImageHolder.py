import os
from PIL import Image


class ImageHolder:
    def __init__(self, root_dir, transform=None, loader=None, iterate=1):
        """
        Args:
            root_dir (str): Root directory path where images are stored in subdirectories.
            transform (callable, optional): A function/transform to apply to the images.
            loader (callable, optional): Function to load an image given its path.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.loader = loader if loader is not None else self.default_loader
        self.samples = []  # List of tuples: (image_path, label)
        self.iterate = iterate

        # Get the class names and corresponding indices
        self.classes, self.class_to_idx = self.find_classes(self.root_dir)
        # Populate the dataset with image paths and labels
        self.make_dataset()

    def find_classes(self, dir):
        """Finds the class folders in a directory.

        Returns:
            A tuple (classes, class_to_idx) where:
                - classes is a list of class names (folder names)
                - class_to_idx is a dict mapping class names to numeric labels
        """
        classes = [entry for entry in os.listdir(dir)
                   if os.path.isdir(os.path.join(dir, entry))]
        classes.sort()
        class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def make_dataset(self):
        """Populates self.samples with (image_path, label) tuples."""
        for class_name in sorted(self.class_to_idx.keys()):
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            # Walk through the directory structure
            for root, _, file_names in sorted(os.walk(class_dir)):
                for file_name in sorted(file_names):
                    if self.is_image_file(file_name):
                        path = os.path.join(root, file_name)
                        label = self.class_to_idx[class_name]
                        self.samples.append((path, label))

    def is_image_file(self, filename):
        """Checks if a file is an image based on its extension."""
        IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
        return filename.lower().endswith(IMG_EXTENSIONS)

    def default_loader(self, path):
        """Default image loader using PIL."""
        return Image.open(path).convert('RGB')

    def __len__(self):
        """Returns the total number of images."""
        return len(self.samples)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, label) where image is loaded (and transformed, if applicable)
        """
        path, label = self.samples[index]
        image = self.loader(path)
        if self.transform:
            images = [self.transform(image) for i in range(self.iterate)]

        labels = [label]*len(images)
        return images, labels
