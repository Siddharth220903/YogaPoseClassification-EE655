# -----------------------------------------------------------------------------
# 1. Download the Yoga Poses Dataset
# -----------------------------------------------------------------------------
import kagglehub
# Fetches the latest version of the dataset from Kaggle
dataset_path = kagglehub.dataset_download("niharika41298/yoga-poses-dataset")
print("Path to dataset files:", dataset_path)


# -----------------------------------------------------------------------------
# 2. Import Required Libraries
# -----------------------------------------------------------------------------
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageFile
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# TensorFlow and Keras for building and training the model
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# -----------------------------------------------------------------------------
# 3. Define Dataset Directories
# -----------------------------------------------------------------------------
# Path where training images are stored
train_dir = '../DATASET/TRAIN'
# Path where testing images are stored
test_dir  = '../DATASET/TEST'


# -----------------------------------------------------------------------------
# 4. Set Up Data Augmentation & Preprocessing
# -----------------------------------------------------------------------------
# For training: apply random width shifts and horizontal flips, scale pixel values
train_datagen = ImageDataGenerator(
    width_shift_range=0.1,
    horizontal_flip=True,
    rescale=1./255,
    validation_split=0.2  # reserve 20% of data for validation
)
# For validation/testing: only scale pixel values
test_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)


# -----------------------------------------------------------------------------
# 5. Create Data Generators
# -----------------------------------------------------------------------------
# Generator for the training subset
train_generator = train_datagen.flow_from_directory(
    directory=train_dir,
    target_size=(75, 75),      # resize all images to 75×75
    color_mode='rgb',          # use RGB channels
    class_mode='categorical',  # for multi-class classification
    batch_size=16,             # number of images per batch
    subset='training'          # select the training split
)

# Generator for the validation subset
validation_generator = test_datagen.flow_from_directory(
    directory=test_dir,
    target_size=(75, 75),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=16,
    subset='validation'        # select the validation split
)


# -----------------------------------------------------------------------------
# 6. Build the CNN Model Architecture
# -----------------------------------------------------------------------------
model = tf.keras.models.Sequential([
    # First convolution block
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                           input_shape=(75, 75, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.25),

    # Second convolution block
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.25),

    # Third convolution block
    tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.25),

    # Flatten the 3D feature maps to 1D feature vectors
    tf.keras.layers.Flatten(),

    # Fully connected layer
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dropout(0.5),

    # Output layer: 5 classes (downdog, goddess, plank, tree, warrior2)
    tf.keras.layers.Dense(5, activation='softmax')
])


# -----------------------------------------------------------------------------
# 7. Compile the Model
# -----------------------------------------------------------------------------
# Use Adam optimizer with a moderate learning rate
optimizer = Adam(learning_rate=0.001)

model.compile(
    loss='categorical_crossentropy',  # multi-class log loss
    optimizer=optimizer,
    metrics=['accuracy']              # track accuracy during training
)

# Training hyperparameters
epochs = 50
batch_size = 16


# -----------------------------------------------------------------------------
# 8. Model Summary
# -----------------------------------------------------------------------------
# Prints a layer-by-layer breakdown of the model’s parameters
model.summary()


# -----------------------------------------------------------------------------
# 9. Handle Truncated Images
# -----------------------------------------------------------------------------
# Allows PIL to load images that may be truncated on disk
ImageFile.LOAD_TRUNCATED_IMAGES = True


# -----------------------------------------------------------------------------
# 10. Train the Model
# -----------------------------------------------------------------------------
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator
)


# -----------------------------------------------------------------------------
# 11. Plot Training & Validation Metrics
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(12, 4))

# Accuracy over epochs
ax[0].plot(history.history['accuracy'], label='Train')
ax[0].plot(history.history['val_accuracy'], label='Validation')
ax[0].set_title('Training vs. Validation Accuracy')
ax[0].set_ylabel('Accuracy')
ax[0].set_xlabel('Epoch')
ax[0].legend(loc='upper left')

# Loss over epochs
ax[1].plot(history.history['loss'], label='Train')
ax[1].plot(history.history['val_loss'], label='Validation')
ax[1].set_title('Training vs. Validation Loss')
ax[1].set_ylabel('Loss')
ax[1].set_xlabel('Epoch')
ax[1].legend(loc='upper left')

# Save the figure at high resolution
plt.savefig("train_test_metrics.png", dpi=300, bbox_inches='tight')
plt.show()


# -----------------------------------------------------------------------------
# 12. Compute & Display Confusion Matrix (Example)
# -----------------------------------------------------------------------------
# NOTE: Replace the placeholders below with your actual labels
y_true = [...]   # e.g., ['tree', 'plank', 'tree', 'warrior2', ...]
y_pred = [...]   # e.g., model predictions mapped to label names

class_names = ['downdog', 'goddess', 'plank', 'tree', 'warrior2']

cm = confusion_matrix(y_true, y_pred, labels=class_names)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# -----------------------------------------------------------------------------
# 13. Save Model Weights
# -----------------------------------------------------------------------------
weights_filename = "cnn_yoga_trained.weights.h5"
model.save_weights(weights_filename)
print(f"Weights saved successfully to {weights_filename}")


# -----------------------------------------------------------------------------
# 14. (Optional) Verify Model Summary Again
# -----------------------------------------------------------------------------
model.summary()


# -----------------------------------------------------------------------------
# 15. Prepare Test Generator for Inference
# -----------------------------------------------------------------------------
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(75, 75),
    batch_size=1,
    class_mode=None,  # no ground-truth labels here
    shuffle=False
)


# -----------------------------------------------------------------------------
# 16. Visualize a Few Test Predictions
# -----------------------------------------------------------------------------
import matplotlib.image as mpimg

filenames = test_generator.filenames
# Populate this list using: predicted_labels = model.predict(test_generator) → argmax → map to class names
predicted_labels = [...]

for i in range(min(5, len(filenames))):
    img_path = os.path.join(test_dir, filenames[i])
    img = mpimg.imread(img_path)
    plt.imshow(img)
    plt.title(f"Predicted: {predicted_labels[i]}")
    plt.axis('off')
    plt.show()


# -----------------------------------------------------------------------------
# 17. Plot an Orange Confusion Matrix (Alternative Visualization)
# -----------------------------------------------------------------------------
# If you prefer an orange color scheme:
cm_alt = confusion_matrix(y_true, y_pred, labels=class_names)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_alt, annot=True, fmt='d', cmap='Oranges',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix (Orange Hue)')
plt.show()
