import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

IMAGE_SIZE = (64, 64)
MODEL_PATH = "SAR_classifier_model_v2.keras"
TEST_IMAGE_PATH = "test1.jpg"

CLASS_NAMES = [
    "buildings",
    "plane",
    "trees",
    "water"
]


if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)


if not os.path.exists(TEST_IMAGE_PATH):
    raise FileNotFoundError(f"Image not found: {TEST_IMAGE_PATH}")

img = image.load_img(TEST_IMAGE_PATH, target_size=IMAGE_SIZE)
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)


predictions = model.predict(img_array)
predicted_index = int(np.argmax(predictions[0]))
confidence = float(np.max(predictions[0]))

predicted_class = CLASS_NAMES[predicted_index]

print(f"Predicted class: {predicted_class}")
print(f"Confidence: {confidence:.4f}")
print(f"Raw probabilities: {predictions[0]}")