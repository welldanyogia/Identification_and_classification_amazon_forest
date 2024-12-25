from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

def create_resnet_model(input_shape, num_classes):
    # Load the ResNet50 base model
    base_model = ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)
    print(f"Base model output type: {type(base_model.output)}")  # Debugging

    # Add pooling and dense layers on top
    x = base_model.output
    print(f"x type after base_model.output: {type(x)}")  # Debugging

    x = GlobalAveragePooling2D()(x)  # Pool across spatial dimensions
    print(f"x type after GlobalAveragePooling2D: {type(x)}")  # Debugging

    x = Dense(128, activation="relu")(x)
    predictions = Dense(num_classes, activation="softmax")(x)

    # Define the final model
    model = Model(inputs=base_model.input, outputs=predictions)

    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False

    return model
