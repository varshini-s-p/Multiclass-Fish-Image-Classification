import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import matplotlib.pyplot as plt
import numpy as np
from data.data_loader import train_generator  # ✅ import train_generator from data_loader

# Function to visualize a batch of augmented images
def visualize_augmented_images():
    class_labels = list(train_generator.class_indices.keys())

    # Get a batch of images and labels
    images, labels = next(train_generator)

    # Plot the first 9 images
    plt.figure(figsize=(12, 12))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i])
        plt.title(f"Class: {class_labels[np.argmax(labels[i])]}")
        plt.axis('off')

    plt.suptitle("Sample Augmented Images from Training Set", fontsize=16)
    plt.tight_layout()
    plt.show()

# Call the function
if __name__ == "__main__":
    visualize_augmented_images()
   

