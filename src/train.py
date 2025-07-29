import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

def build_model(input_shape=(224, 224, 1), num_classes=14):
    """
    Build a transfer learning model using ResNet50 for multi-label classification.

    Args:
        input_shape (tuple): Shape of input images.
        num_classes (int): Number of disease labels.

    Returns:
        tf.keras.Model: Compiled Keras model.
    """
    # Create input layer
    input_tensor = Input(shape=input_shape)
    
    # Convert grayscale to RGB by replicating the channel
    x = tf.keras.layers.Lambda(lambda x: tf.repeat(x, 3, axis=-1))(input_tensor)

    # Load ResNet50 base
    base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=x)
    base_model.trainable = False  # freeze layers

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(0.5)(x)
    output = Dense(num_classes, activation='sigmoid')(x)

    model = Model(inputs=input_tensor, outputs=output)

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )

    return model


def train_model(X_train, y_train, X_val, y_val, num_classes, batch_size=32, epochs=20):
    """
    Train the ResNet50 model.

    Args:
        X_train (np.ndarray): Preprocessed training images.
        y_train (np.ndarray): Multi-label encoded training labels.
        X_val (np.ndarray): Validation images.
        y_val (np.ndarray): Validation labels.
        num_classes (int): Number of output labels.

    Returns:
        tf.keras.Model: Trained model
    """
    model = build_model(input_shape=X_train.shape[1:], num_classes=num_classes)

    os.makedirs("outputs", exist_ok=True)

    checkpoint = ModelCheckpoint(
        "outputs/model_weights.h5",
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    earlystop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[checkpoint, earlystop]
    )

    return model, history
