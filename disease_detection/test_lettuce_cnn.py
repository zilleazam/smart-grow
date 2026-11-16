"""Run the trained lettuce CNN on a single image and show the prediction.

Usage: python test_lettuce_cnn.py <image_path>
"""
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2
from tensorflow import keras

MODEL_PATH = "models/trained_lettuce_model.keras"
TEST_DIR = "data/lettuce_dataset/test"  # used only to read the class names
image_path = sys.argv[1] if len(sys.argv) > 1 else "test_image.jpg"

# Load the model
model = keras.models.load_model(MODEL_PATH)

# Define class names
test_set = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    labels="inferred",
    label_mode="categorical",
    batch_size=32,
    image_size=(128, 128),
    shuffle=False,
    seed=123
)
class_names = test_set.class_names  # Get class labels

# Load and preprocess the test image
image = tf.keras.preprocessing.image.load_img(image_path, target_size=(128, 128))
input_arr = tf.keras.preprocessing.image.img_to_array(image)
input_arr = np.expand_dims(input_arr, axis=0)  # Add batch dimension
img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.title('Test Image')
plt.xticks([])
plt.yticks([])
plt.show()

# Make predictions
predictions = model.predict(input_arr)

# Get predicted class index and probability
result_index = np.argmax(predictions)
predicted_class = class_names[result_index]
prediction_confidence = predictions[0][result_index] * 100  # Convert to percentage

# Display the original image with the result
plt.imshow(img)
plt.title(f"Disease Name: {predicted_class} ({prediction_confidence:.2f}%)")
plt.xticks([])
plt.yticks([])
plt.show()

# Print detailed probabilities for all classes
print("Prediction Probabilities:")
for i, class_name in enumerate(class_names):
    print(f"{class_name}: {predictions[0][i] * 100:.2f}%")
