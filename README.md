#  Multiclass Fish Image Classification

This project focuses on classifying fish images into **11 distinct species** using deep learning models. The goal is to automate species identification using a robust image classification pipeline.

---

##  Project Overview

This deep learning-based classifier takes an image of a fish as input and predicts its species using a fine-tuned **EfficientNetB0** model. Several CNN architectures were tested, and the best-performing model was selected based on accuracy and generalization.

---

##  Model Highlights

- **Best Performing Model**: EfficientNetB0 (pre-trained on ImageNet)
- **Input Size**: 224x224
- **Output**: 11-class softmax layer
- **Loss Function**: Categorical Crossentropy
- **Optimizer**: Adam
- **Regularization**: Data Augmentation, Dropout
- **Training Techniques**:
  - Early Stopping
  - Model Checkpointing
  - Transfer Learning
  - Fine-tuning of final layers

---

##  Models Compared

| Model           | Accuracy |
|----------------|----------|
| CNN (from scratch)     | ~70%     |
| VGG16           | ~84%     |
| ResNet50        | ~86%     |
| MobileNetV2     | ~88%     |
| InceptionV3     | ~89%     |
| **EfficientNetB0** | **91.3%** |

> EfficientNetB0 gave the **highest validation accuracy** with a good balance of performance and efficiency.

---

##  Dataset Overview

- Total Classes: 11
- Fish species include:
  - Fish, Sea Bass, Shrimp, Trout, Red Mullet, Hourse Mackerel, Gilt-Head Bream, etc.
- Dataset was split as:
  - **Train**: 70%
  - **Validation**: 15%
  - **Test**: 15%

> Note: Due to GitHub file size constraints, the dataset is not included here.

---

## ⚙️ Project Structure


