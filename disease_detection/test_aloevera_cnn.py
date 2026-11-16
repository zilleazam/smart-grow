"""Run the trained aloe vera CNN on a single image and show the prediction.

Usage: python test_aloevera_cnn.py <image_path>
"""
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2
from tensorflow import keras

MODEL_PATH = "models/trained_aloevera_disease_detect_model.keras"
TEST_DIR = "data/aloe_vera_dataset/test"  # used for class names and batch evaluation
image_path = sys.argv[1] if len(sys.argv) > 1 else "test_image.jpg"

# Load the model
model = keras.models.load_model(MODEL_PATH)

test_set = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    labels="inferred",
    label_mode="categorical",
    batch_size=32,
    image_size=(128, 128),
    shuffle=False,
    seed=123
)
class_names = test_set.class_names

y_pred = model.predict(test_set)
predicted_categories = tf.argmax(y_pred, axis=1)
true_categories = tf.concat([y for x, y in test_set], axis=0)
Y_true = tf.argmax(true_categories, axis=1)

# Test Image Visualization
img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Converting BGR to RGB
plt.imshow(img)
plt.title('Test Image')
plt.xticks([])
plt.yticks([])
plt.show()

image = tf.keras.preprocessing.image.load_img(image_path, target_size=(128, 128))
input_arr = tf.keras.preprocessing.image.img_to_array(image)
input_arr = np.array([input_arr])  # Convert single image to a batch
predictions = model.predict(input_arr)

# Display predictions
print(predictions)
result_index = np.argmax(predictions)  # Return index of max element
print(result_index)

# Displaying the disease prediction
model_prediction = class_names[result_index]
plt.imshow(img)
plt.title(f"Plant State: {model_prediction}")
plt.xticks([])
plt.yticks([])
plt.show()
