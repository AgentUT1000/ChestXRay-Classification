import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

def compute_gradcam(model, image, class_index, layer_name="conv5_block3_out"):
    """
    Compute Grad-CAM heatmap for a specific class on a single image.

    Args:
        model (tf.keras.Model): Trained model.
        image (np.ndarray): Input image of shape (224, 224, 1).
        class_index (int): Index of the target class.
        layer_name (str): Layer to compute Grad-CAM from.

    Returns:
        np.ndarray: Heatmap of shape (224, 224)
    """
    # Ensure image is in the correct format for the model
    if image.ndim == 3 and image.shape[-1] == 1:
        # Image is already grayscale, which is what the model expects
        pass
    else:
        # If image has 3 channels, convert to grayscale
        if image.ndim == 3 and image.shape[-1] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            image = np.expand_dims(image, axis=-1)
    
    # Normalize to [0, 1] range and convert to float32
    if image.dtype != np.float32:
        image = image.astype(np.float32)
    if image.max() > 1.0:
        image = image / 255.0

    # Create input tensor with proper shape (224, 224, 1)
    img_tensor = tf.expand_dims(image, axis=0)

    # Create gradient model
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= tf.math.reduce_max(heatmap) + 1e-8
    heatmap = heatmap.numpy()
    heatmap = cv2.resize(heatmap, (224, 224))

    return heatmap


def overlay_gradcam(image, heatmap, alpha=0.4, cmap='jet'):
    """
    Overlay heatmap onto original image.

    Args:
        image (np.ndarray): Grayscale input image (224x224x1 or 224x224).
        heatmap (np.ndarray): Grad-CAM heatmap.
        alpha (float): Transparency for overlay.
        cmap (str): Colormap to apply.

    Returns:
        np.ndarray: RGB overlay image
    """
    if image.ndim == 3:
        image = image.squeeze()

    # Ensure image is in [0, 255] range for OpenCV
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)

    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    overlayed = cv2.addWeighted(image_rgb, 1 - alpha, heatmap_colored, alpha, 0)
    return overlayed


def generate_and_save_gradcam(model, image_path, class_index, class_name, preprocess_fn):
    """
    Process an image, compute Grad-CAM, and save overlay.

    Args:
        model: Trained model
        image_path (str): Path to the image
        class_index (int): Target class index
        class_name (str): Human-readable label name
        preprocess_fn (func): Your preprocessing pipeline
    """
    os.makedirs("outputs/plots", exist_ok=True)

    # Check if the default layer exists, otherwise use a fallback
    layer_name = "conv5_block3_out"
    try:
        model.get_layer(layer_name)
    except ValueError:
        # Try to find a suitable layer for Grad-CAM
        available_layers = [layer.name for layer in model.layers if 'conv' in layer.name.lower()]
        if available_layers:
            layer_name = available_layers[-1]  # Use the last convolutional layer
            print(f"[WARNING] Layer 'conv5_block3_out' not found. Using '{layer_name}' instead.")
        else:
            print("[ERROR] No suitable convolutional layer found for Grad-CAM. Skipping...")
            return

    image = preprocess_fn(image_path)
    heatmap = compute_gradcam(model, image, class_index, layer_name)
    overlay = overlay_gradcam(image, heatmap)

    # Save
    filename = os.path.basename(image_path).replace(".png", "")
    out_path = f"outputs/plots/gradcam_{filename}_{class_name}.png"
    cv2.imwrite(out_path, overlay)
    print(f"[✓] Saved Grad-CAM: {out_path}")
