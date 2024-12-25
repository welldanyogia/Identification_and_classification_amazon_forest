import os

from tensorflow.keras.applications import MobileNet
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

from models.mobilenet_model import create_mobilenet_model


def train_mobilenet(input_shape=(224, 224, 3), epochs=15, batch_size=32):
    # Data generators for training and validation
    from keras.src.legacy.preprocessing.image import ImageDataGenerator
    train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    train_data = train_datagen.flow_from_directory(
        'data/planets/train-jpg',
        target_size=input_shape[:2],
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )
    val_data = train_datagen.flow_from_directory(
        'data/planets/train-jpg',
        target_size=input_shape[:2],
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    # Dynamically set num_classes
    num_classes = len(os.listdir('data/planets/train-jpg'))

    # Load MobileNet base model
    mobilenet_base = MobileNet(weights="imagenet", include_top=False, input_shape=input_shape)

    # Add custom classification layers
    x = GlobalAveragePooling2D()(mobilenet_base.output)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    output = Dense(num_classes, activation="softmax")(x)

    model = create_mobilenet_model(input_shape=mobilenet_base.input, num_classes=num_classes)

    # Freeze the base layers
    for layer in mobilenet_base.layers:
        layer.trainable = False

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss="categorical_crossentropy", metrics=["accuracy"])

    # Display the model summary
    print("MobileNet Model Summary:")
    model.summary()

    # Train the model
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs
    )

    # Simpan model setelah pelatihan
    model.save('mobilenet_model.h5')  # Gunakan .h5 atau direktori sesuai kebutuhan
    print("Model telah disimpan ke file mobilenet_model.h5")

    # Return the trained model and history
    return model, history
