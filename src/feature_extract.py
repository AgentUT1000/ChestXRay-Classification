import cv2
import numpy as np
import os

def extract_descriptors(image, method="SIFT"):
    """
    Extract keypoint descriptors from a preprocessed grayscale image.

    Args:
        image (np.ndarray): Input image of shape (H, W) or (H, W, 1)
        method (str): "SIFT" or "ORB"

    Returns:
        np.ndarray or None: Descriptor array of shape (num_keypoints, feature_dim)
    """
    if image.ndim == 3:
        image = image.squeeze()

    if method.upper() == "SIFT":
        detector = cv2.SIFT_create()
    elif method.upper() == "ORB":
        detector = cv2.ORB_create()
    else:
        raise ValueError("Unsupported method. Choose 'SIFT' or 'ORB'.")

    keypoints, descriptors = detector.detectAndCompute(image, None)
    return descriptors  # Can be None if no keypoints found


def extract_descriptors_batch(image_paths, preprocess_fn, method="SIFT"):
    """
    Extract descriptors for a batch of images.

    Args:
        image_paths (list of str): Paths to image files.
        preprocess_fn (function): Image preprocessing function.
        method (str): "SIFT" or "ORB".

    Returns:
        dict: {image_id: descriptors or None}
    """
    descriptor_map = {}

    for path in image_paths:
        try:
            img = preprocess_fn(path)
            desc = extract_descriptors(img, method=method)
            image_id = os.path.basename(path)
            descriptor_map[image_id] = desc
        except Exception as e:
            print(f"[ERROR] {path}: {e}")
            descriptor_map[image_id] = None

    return descriptor_map


def save_descriptors(descriptor_dict, out_path):
    """
    Save descriptor dictionary to a .npy file.

    Args:
        descriptor_dict (dict): Map of {image_id: descriptors}
        out_path (str): Output file path
    """
    np.save(out_path, descriptor_dict)
