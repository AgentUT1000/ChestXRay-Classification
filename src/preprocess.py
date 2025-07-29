import os
import cv2
import numpy as np

def preprocess_image(image_path):
    # Ensure cross-platform compatibility
    image_path = os.path.normpath(image_path)

    # Load grayscale image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"[WARN] Skipping unreadable or missing image: {image_path}")
        return None

    # Resize to 224x224
    image = cv2.resize(image, (224, 224))

    # Add Gaussian noise
    noise = np.random.normal(0, 5, image.shape).astype(np.uint8)
    noisy = cv2.add(image, noise)

    # Histogram Equalization
    equalized = cv2.equalizeHist(noisy)

    return np.expand_dims(equalized, axis=-1)
