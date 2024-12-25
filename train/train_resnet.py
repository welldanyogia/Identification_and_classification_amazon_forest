import os
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

def create_resnet_model(input_shape, num_classes):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

def compile_and_train(model, train_data, val_data, epochs, batch_size):
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        batch_size=batch_size
    )
    return history

def train_resnet(epochs=10, batch_size=32):
    input_shape = (224, 224, 3)
    num_classes = len(os.listdir('data/planets/train-jpg'))  # Dinamis berdasarkan dataset

    model = create_resnet_model(input_shape, num_classes)

    train_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
        'data/planets/train-jpg',
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical'
    )
    val_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
        'data/planets/train-jpg',
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical'
    )

    history = compile_and_train(
        model=model,
        train_data=train_gen,
        val_data=val_gen,
        epochs=epochs,
        batch_size=batch_size
    )

    # Simpan model setelah pelatihan
    model.save('resnet_model.h5')  # Simpan model ke file dengan format .h5
    print("Model telah disimpan ke file resnet_model.h5")

    return model, history
