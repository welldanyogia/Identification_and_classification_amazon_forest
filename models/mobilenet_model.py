from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint


def create_mobilenet_model(input_shape, num_classes):
    """
    Membangun model MobileNet dari nol.
    """
    model = Sequential()

    # Convolutional Layer (Initial)
    model.add(Conv2D(32, (3, 3), strides=(2, 2), padding="same", input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(ReLU())

    # Depthwise Separable Convolutions
    def depthwise_separable_block(filters, stride):
        model.add(DepthwiseConv2D((3, 3), padding="same", strides=stride))
        model.add(BatchNormalization())
        model.add(ReLU())
        model.add(Conv2D(filters, (1, 1), padding="same"))
        model.add(BatchNormalization())
        model.add(ReLU())

    # Add Depthwise Blocks
    depthwise_separable_block(64, (1, 1))
    depthwise_separable_block(128, (2, 2))
    depthwise_separable_block(128, (1, 1))
    depthwise_separable_block(256, (2, 2))
    depthwise_separable_block(256, (1, 1))
    depthwise_separable_block(512, (2, 2))

    for _ in range(5):  # 5 Depthwise blocks with stride 1
        depthwise_separable_block(512, (1, 1))

    depthwise_separable_block(1024, (2, 2))
    depthwise_separable_block(1024, (1, 1))

    # Global Average Pooling and Dense Output
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation="softmax"))

    return model


def compile_and_train_mobilenet(input_shape, num_classes, train_data, val_data, epochs=15, batch_size=32):
    """
    Fungsi untuk mengompilasi dan melatih MobileNet dari nol.
    """
    model = create_mobilenet_model(input_shape, num_classes)

    # Compile Model
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    # Callback untuk menyimpan model terbaik
    checkpoint = ModelCheckpoint("best_mobilenet_model.keras", monitor="val_loss", save_best_only=True, verbose=1)

    # Train Model
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[checkpoint]
    )

    return model, history


def train_mobilenet_from_scratch(epochs=15, batch_size=32, train_dir="data/planets/train-jpg", val_dir="data/planets/train-jpg"):
    """
    Fungsi untuk melatih MobileNet dari awal dengan data.
    """
    input_shape = (224, 224, 3)

    # ImageDataGenerator
    train_datagen = ImageDataGenerator(rescale=1./255)
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_data = train_datagen.flow_from_directory(
        train_dir,
        target_size=input_shape[:2],
        batch_size=batch_size,
        class_mode='categorical'
    )

    val_data = val_datagen.flow_from_directory(
        val_dir,
        target_size=input_shape[:2],
        batch_size=batch_size,
        class_mode='categorical'
    )

    # Determine the number of classes dynamically
    num_classes = len(train_data.class_indices)

    # Compile and train model
    model, history = compile_and_train_mobilenet(input_shape, num_classes, train_data, val_data, epochs, batch_size)

    return model, history
