"""Train the CNN disease-detection model for lettuce (4 classes).

Classes: Bacterial Anthracnose, Dead Lettuce, Fungal Downy Mildew, Healthy Lettuce.
Originally run on Google Colab with the dataset on Google Drive.
"""
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense,
                                     Dropout, BatchNormalization, SpatialDropout2D)
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

DATASET_DIR = "/content/drive/MyDrive/lettuce_dataset"
TEST_IMAGE_PATH = "/content/drive/MyDrive/test_image.jpg"  # replace with your own image

# Load Training Data
training_set = tf.keras.utils.image_dataset_from_directory(
    f"{DATASET_DIR}/train",
    labels="inferred",
    label_mode="categorical",
    batch_size=32,
    image_size=(128, 128),
    shuffle=True,
    seed=123
)

# Load Validation Data
validation_set = tf.keras.utils.image_dataset_from_directory(
    f"{DATASET_DIR}/valid",
    labels="inferred",
    label_mode="categorical",
    batch_size=32,
    image_size=(128, 128),
    shuffle=True,
    seed=123
)

# Define CNN model
model = Sequential([
    # First Conv Block
    Conv2D(64, (5, 5), activation='relu', padding='same', input_shape=(128, 128, 3)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.2),

    # Second Conv Block
    Conv2D(128, (5, 5), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    SpatialDropout2D(0.3),

    # Third Conv Block
    Conv2D(256, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    SpatialDropout2D(0.3),

    # Fourth Conv Block (Added for More Feature Extraction)
    Conv2D(512, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    SpatialDropout2D(0.3),

    # Fully Connected Layers
    Flatten(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),

    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),

    Dense(4, activation='softmax')  # Output for 4 classes
])

optimizer = Adam(learning_rate=0.0005)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Learning Rate Scheduler
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(training_set, epochs=30, validation_data=validation_set,
                    callbacks=[lr_scheduler, early_stopping])

model.save("trained_lettuce_model.keras")

# Load Test Data
test_set = tf.keras.utils.image_dataset_from_directory(
    f"{DATASET_DIR}/test",
    labels="inferred",
    label_mode="categorical",
    batch_size=32,
    image_size=(128, 128),
    shuffle=False,
    seed=123
)
class_names = test_set.class_names

# Predictions
y_pred = model.predict(test_set)
predicted_categories = tf.argmax(y_pred, axis=1)
true_categories = tf.concat([y for x, y in test_set], axis=0)
Y_true = tf.argmax(true_categories, axis=1)

# Test Image Prediction
img = cv2.imread(TEST_IMAGE_PATH)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.title('Test Image')
plt.xticks([])
plt.yticks([])
plt.show()

image = tf.keras.preprocessing.image.load_img(TEST_IMAGE_PATH, target_size=(128, 128))
input_arr = tf.keras.preprocessing.image.img_to_array(image)
input_arr = np.array([input_arr])

predictions = model.predict(input_arr)

# Get the predicted class index and probability
result_index = np.argmax(predictions)
predicted_class = class_names[result_index]
prediction_confidence = predictions[0][result_index] * 100  # Convert to percentage

# Display the image with prediction results
plt.imshow(img)
plt.title(f"Disease Name: {predicted_class} ({prediction_confidence:.2f}%)")
plt.xticks([])
plt.yticks([])
plt.show()

# Print detailed probabilities for all classes
for i, class_name in enumerate(class_names):
    print(f"{class_name}: {predictions[0][i] * 100:.2f}%")
