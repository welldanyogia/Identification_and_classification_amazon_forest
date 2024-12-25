from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data(data_dir, img_size=(150, 150), batch_size=32):
    datagen = ImageDataGenerator(rescale=1.0 / 255.0, validation_split=0.2)

    train_data = datagen.flow_from_directory(
        f"{data_dir}/train-jpg",
        target_size=img_size,
        batch_size=batch_size,
        subset="training",
        class_mode="categorical",
    )

    val_data = datagen.flow_from_directory(
        f"{data_dir}/train-jpg",
        target_size=img_size,
        batch_size=batch_size,
        subset="validation",
        class_mode="categorical",
    )

    test_data = datagen.flow_from_directory(
        f"{data_dir}/test-jpg",
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
    )

    return train_data, val_data, test_data
