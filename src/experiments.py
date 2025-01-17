from data_augmentations import RepeatImageTransform
from PIL import Image
import matplotlib.pyplot as plt




# Load a 128x128 example image
image = Image.open(r'D:\vscode\DeepLearning2425\dl2024-competition-notbad\dataset\test\2\image_0081.jpg')  # Replace with your image path

# Define the transform (repeat 2x along height)
repeat_transform = RepeatImageTransform()

# Apply the transform
repeated_image = repeat_transform(image)

# Visualize the result
plt.imshow(repeated_image)
plt.title("Repeated Image (256x128)")
plt.show()