"""Train the CNN disease-detection model for aloe vera (3 classes).

Classes: Fresh Aloe Vera, Rot Aloe Vera, Rust Aloe Vera.
Originally run on Google Colab with the dataset on Google Drive.
"""
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix

DATASET_DIR = "/content/drive/MyDrive/aloe_vera_dataset"
MODEL_PATH = "/content/drive/MyDrive/new_trained_aloevera_model.keras"
TEST_IMAGE_PATH = f"{DATASET_DIR}/test/Rot Augmented Aloe Vera/AloeVeraOriginalRot0014_bright_1862.jpg"

training_set = tf.keras.utils.image_dataset_from_directory(
    f"{DATASET_DIR}/train",
    labels="inferred",
    label_mode="categorical",
    batch_size=32,
    image_size=(128, 128),
    shuffle=False,
    seed=123
)

validation_set = tf.keras.utils.image_dataset_from_directory(
    f"{DATASET_DIR}/valid",
    labels="inferred",
    label_mode="categorical",
    batch_size=32,
    image_size=(128, 128),
    shuffle=False,
    seed=123
)

print("Training set:", training_set.cardinality().numpy(), "batches")
print("Validation set:", validation_set.cardinality().numpy(), "batches")

for images, labels in training_set.take(1):
    print("Image shape:", images.shape)
    print("Labels shape:", labels.shape)

cnn = tf.keras.models.Sequential([
    # First Conv Block
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),

    # Second Conv Block
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    # Third Conv Block
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    # Flattening
    tf.keras.layers.Flatten(),

    # Fully Connected Layers
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.3),  # Helps prevent overfitting

    # Output Layer (3 classes, softmax activation)
    tf.keras.layers.Dense(3, activation='softmax')
])

cnn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy'])

training_history = cnn.fit(x=training_set, validation_data=validation_set, epochs=30)

# Training set Accuracy
train_loss, train_acc = cnn.evaluate(training_set)
print('Training accuracy:', train_acc)

# Validation set Accuracy
val_loss, val_acc = cnn.evaluate(validation_set)
print('Validation accuracy:', val_acc)

# Plot training and validation performance
plt.plot(training_history.history['accuracy'], label='Training Accuracy')
plt.plot(training_history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title("Training and Validation Performance")
plt.show()

cnn.save(MODEL_PATH)

# Load the saved model
cnn = load_model(MODEL_PATH)
cnn.summary()

for layer in cnn.layers:
    weights = layer.get_weights()
    print(f"Layer {layer.name} has {len(weights)} weight matrices")

test_set = tf.keras.utils.image_dataset_from_directory(
    f"{DATASET_DIR}/test",
    labels="inferred",
    label_mode="categorical",
    batch_size=32,
    image_size=(128, 128),
    shuffle=False,
    seed=123
)
class_name = test_set.class_names

y_pred = cnn.predict(test_set)
predicted_categories = tf.argmax(y_pred, axis=1)

true_categories = tf.concat([y for x, y in test_set], axis=0)
Y_true = tf.argmax(true_categories, axis=1)

# Classification Report
print("Classification Report:")
print(classification_report(Y_true, predicted_categories, target_names=test_set.class_names))

# Confusion Matrix
cm = confusion_matrix(Y_true, predicted_categories)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=test_set.class_names,
            yticklabels=test_set.class_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Test Image Visualization
img = cv2.imread(TEST_IMAGE_PATH)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Converting BGR to RGB
plt.imshow(img)
plt.title('Test Image')
plt.xticks([])
plt.yticks([])
plt.show()

image = tf.keras.preprocessing.image.load_img(TEST_IMAGE_PATH, target_size=(128, 128))
input_arr = tf.keras.preprocessing.image.img_to_array(image)
input_arr = np.array([input_arr])  # Convert single image to a batch
predictions = cnn.predict(input_arr)

# Display predictions
print(predictions)
result_index = np.argmax(predictions)  # Return index of max element
print(result_index)

# Displaying the disease prediction
model_prediction = class_name[result_index]
plt.imshow(img)
plt.title(f"Disease Name: {model_prediction}")
plt.xticks([])
plt.yticks([])
plt.show()
