import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

def compile_and_train(
    model,
    train_data,
    val_data,
    epochs=10,
    batch_size=32,
    optimizer=None,
    loss="categorical_crossentropy",
    metrics=["accuracy"]
):
    if optimizer is None:
        optimizer = tf.keras.optimizers.Adam()

    # Compile the model
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # Train the model
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[
            ModelCheckpoint("best_model.keras", monitor="val_loss", save_best_only=True, verbose=1),
            EarlyStopping(monitor="val_loss", patience=3, verbose=1)
        ]
    )

    return history


def plot_training_history(history):
    """
    Function to plot the training and validation accuracy and loss curves.
    """
    # Plot accuracy
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()

    plt.tight_layout()
    plt.show()
