import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint

# Mengonversi label ke one-hot encoding
from tensorflow.keras.utils import to_categorical

def build_model(input_shape, num_classes=449):
    """
    Fungsi untuk membangun model CNN.
    """
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')  # Output untuk 449 kelas
    ])
    return model

def compile_and_train(model, train_data, val_data, epochs, batch_size, num_classes=449):
    """
    Fungsi untuk mengkompilasi dan melatih model.
    """
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',  # Loss untuk klasifikasi multikelas
                  metrics=['accuracy'])

    # Callback untuk menyimpan model terbaik
    checkpoint = ModelCheckpoint("best_model.keras", monitor="val_loss", save_best_only=True, verbose=1)

    history = model.fit(
        train_data,
        epochs=epochs,
        validation_data=val_data,
        batch_size=batch_size,
        callbacks=[checkpoint]
    )
    return history

def train_cnn(epochs=15, batch_size=32, train_dir="data/planets/train-jpg", val_dir="data/planets/train-jpg"):
    """
    Fungsi untuk melatih model CNN dengan data yang diberikan.
    """
    input_shape = (224, 224, 3)  # Sesuaikan dengan ukuran gambar input Anda
    num_classes = 449  # Jumlah kelas yang sesuai dengan dataset Anda

    # Menggunakan ImageDataGenerator untuk memuat data gambar
    train_datagen = ImageDataGenerator(rescale=1./255)
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_data = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',  # Pastikan label dalam format one-hot
    )

    val_data = val_datagen.flow_from_directory(
        val_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',  # Pastikan label dalam format one-hot
    )

    # Membangun model
    model = build_model(input_shape, num_classes)

    # Melatih model
    history = compile_and_train(model, train_data, val_data, epochs, batch_size, num_classes)

    return model, history
