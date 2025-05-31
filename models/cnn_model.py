# models/cnn_model.py

import os
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import sys

#  Add parent folder to system path for data import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.data_loader import train_generator, val_generator, test_generator

#  Step 1: Define CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
   MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(train_generator.num_classes, activation='softmax')  # dynamically detects number of species
])

#  Step 2: Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#  Step 3: Model Checkpoint (save best model)
os.makedirs('models', exist_ok=True)
checkpoint_path = 'models/cnn_model.h5'
checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True, mode='max')

#  Step 4: Train model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    callbacks=[checkpoint]
)

#  Step 5: Save training curves
def plot_training(history):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('models/training_curves.png')
    plt.show()

plot_training(history)

#  Step 6: Save training history
with open('models/training_history.pkl', 'wb') as f:
    pickle.dump(history.history, f)
import sys
import os
import numpy as np
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model  # if you're loading a saved model

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.data_loader import test_generator


# Load CNN model (adjust the path if needed)
model = load_model('models/outputs/cnn_model.h5')

#  Step 7: Evaluate on test set
test_loss, test_acc = model.evaluate(test_generator)
print(f"\n  Test Accuracy: {test_acc:.2f}")

#  Step 8: Classification report + confusion matrix
y_pred = model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

print("\n Classification Report:")
print(classification_report(y_true, y_pred_classes, target_names=class_labels))

print("\n Confusion Matrix:")
print(confusion_matrix(y_true, y_pred_classes))


