import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from collections import Counter

from src.preprocess import preprocess_image
from src.train import train_model
from src.evaluate import evaluate_model
from src.gradcam import generate_and_save_gradcam

def load_metadata(csv_path, image_dir):
    df = pd.read_csv(csv_path)
    df['image_path'] = df['Image Index'].apply(lambda x: os.path.normpath(os.path.join(image_dir, x)))

    # Step 2: Check if paths from CSV actually exist
    df['image_exists'] = df['image_path'].apply(lambda path: os.path.exists(os.path.normpath(path)))
    missing = df[~df['image_exists']]
    print(f"[DEBUG] Missing images in CSV: {len(missing)}")
    print(missing[['Image Index', 'image_path']].head())

    # Optional: Log which images were missing (audit)
    if len(missing) > 0:
        os.makedirs('outputs', exist_ok=True)
        missing.to_csv("outputs/missing_images.csv", index=False)

    # Step 3: Hotfix — only keep rows with images that exist
    df = df[df['image_exists']].reset_index(drop=True)
    df['labels'] = df['Finding Labels'].apply(lambda x: x.split('|'))
    return df

def preprocess_images(image_paths):
    print("[INFO] Preprocessing images...")
    processed_images = []
    valid_indices = []

    for i, path in enumerate(image_paths):
        result = preprocess_image(path)
        if result is not None:
            processed_images.append(result)
            valid_indices.append(i)

    return np.array(processed_images), valid_indices

def main():
    # === Paths ===
    CSV_PATH = "data/Ground_Truth.csv"
    IMAGE_DIR = "data/images"
    OUTPUT_DIR = "outputs"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # === Load & process metadata ===
    df = load_metadata(CSV_PATH, IMAGE_DIR)

    # === Initial label encoding (for frequency check only) ===
    mlb = MultiLabelBinarizer()
    df['labels'] = df['labels'].apply(lambda x: list(set(x)))  # remove duplicates
    Y_full = mlb.fit_transform(df['labels'])
    label_counts = Y_full.sum(axis=0)
    rare_classes = np.where(label_counts < 2)[0]
    rare_labels = mlb.classes_[rare_classes]

    # === Filter rows containing rare labels
    def has_rare(labels):
        return any(label in rare_labels for label in labels)
    df_filtered = df[~df['labels'].apply(has_rare)].reset_index(drop=True)

    # === Final label encoding after filtering
    mlb_filtered = MultiLabelBinarizer()
    Y_filtered = mlb_filtered.fit_transform(df_filtered['labels'])
    class_names_filtered = mlb_filtered.classes_
    np.save(os.path.join(OUTPUT_DIR, "class_labels.npy"), class_names_filtered)

    # === Check if stratification is still safe
    filtered_counts = Y_filtered.sum(axis=0)
    stratify_arg = Y_filtered if all(filtered_counts >= 2) else None

    # === Train/Val/Test Split ===
    from skmultilearn.model_selection import iterative_train_test_split

    # Prepare data
    X_all = df_filtered['image_path'].values.reshape(-1, 1)  # shape (n_samples, 1)
    y_all = Y_filtered

    # === Train / Temp Split (80/20)
    X_train, y_train, X_temp, y_temp = iterative_train_test_split(X_all, y_all, test_size=0.2)

    # === Val / Test Split (50/50 of temp → 10/10)
    X_val, y_val, X_test, y_test = iterative_train_test_split(X_temp, y_temp, test_size=0.5)

    # === Build DataFrames back from paths
    train_df = df_filtered[df_filtered['image_path'].isin(X_train.flatten())].reset_index(drop=True)
    val_df   = df_filtered[df_filtered['image_path'].isin(X_val.flatten())].reset_index(drop=True)
    test_df  = df_filtered[df_filtered['image_path'].isin(X_test.flatten())].reset_index(drop=True)

    # === Preprocess all images ===
    X_train, train_indices = preprocess_images(train_df['image_path'])
    X_val, val_indices = preprocess_images(val_df['image_path'])
    X_test, test_indices = preprocess_images(test_df['image_path'])

    # === Train model ===
    model, history = train_model(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        num_classes=len(class_names_filtered)
    )

    # === Evaluate model ===
    metrics = evaluate_model(model, X_test, y_test, class_names_filtered)
    print("[METRICS]", metrics)

    # === Generate Grad-CAMs for first 3 test images and first 3 labels
    print("[INFO] Generating Grad-CAMs...")
    for idx in range(min(3, len(test_df))):
        for cls in range(min(3, len(class_names_filtered))):
            generate_and_save_gradcam(
                model=model,
                image_path=test_df['image_path'].iloc[idx],
                class_index=cls,
                class_name=class_names_filtered[cls],
                preprocess_fn=preprocess_image
            )

if __name__ == "__main__":
    main()
