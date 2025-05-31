import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from data.data_loader import train_generator

# Ensure we can import from the project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def plot_class_distribution():
    # Get class indices (e.g., {'class_name': index})
    class_indices = train_generator.class_indices
    class_labels = list(class_indices.keys())

    # Get actual labels from the generator
    labels = train_generator.classes  # This gives you a list of numeric labels

    # Count how many samples per class
    label_counts = Counter(labels)

    # Map back numeric labels to class names
    class_counts = {class_labels[k]: v for k, v in label_counts.items()}

    # Plotting
    plt.figure(figsize=(12, 6))
    sns.barplot(x=list(class_counts.keys()), y=list(class_counts.values()), palette="viridis")
    plt.title("Class Distribution in Training Set")
    plt.xlabel("Fish Species")
    plt.ylabel("Number of Images")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_class_distribution()

