import tensorflow as tf
import cv2
import numpy as np
from tkinter import Tk, filedialog
import matplotlib.pyplot as plt
import os

# Constants
MODEL_PATH = "best_brain_stroke_model.keras"  # Update if needed
IMG_SIZE = (150, 150)
THRESHOLD = 0.5  # Adjusted threshold after analyzing predictions

# Load the trained model safely
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
    print("[INFO] Model loaded successfully!")
else:
    print(f"[ERROR] Model file '{MODEL_PATH}' not found. Please train the model first.")
    exit(1)


def load_image():
    root = Tk()
    root.withdraw()  # Hide root window
    root.attributes('-topmost', True)  # Keep file dialog on top
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")])
    root.destroy()  # Destroy root window after selection
    return file_path


def predict_brain_stroke(image_path):
    # Load and preprocess the image
    img = cv2.imread(image_path)
    if img is None:
        print("[ERROR] Could not load image.")
        return

    img_resized = cv2.resize(img, IMG_SIZE)
    img_array = np.expand_dims(img_resized, axis=0) / 255.0  # Normalize

    # Make prediction
    prediction = model.predict(img_array)[0][0]
    confidence = prediction * 100 if prediction < 0.5 else (1 - prediction) * 100
    label = "Stroke Detected" if prediction < THRESHOLD else "No Stroke Detected"

    # Convert BGR to RGB for correct color representation
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Show results
    plt.figure(figsize=(6, 6))
    plt.imshow(img_rgb)
    plt.title(f"{label}")
    plt.axis('off')
    plt.show()

    print(f"[INFO] {label} with {confidence:.2f}% confidence.")


# Example Usage
if __name__ == "__main__":
    image_path = load_image()  # Select image using file explorer
    if image_path:
        predict_brain_stroke(image_path)
    else:
        print("[INFO] No image selected.")
