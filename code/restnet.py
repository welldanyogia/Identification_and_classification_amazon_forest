from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Data preparation
datagen = ImageDataGenerator(rescale=1.0/255.0, validation_split=0.2)

train_data = datagen.flow_from_directory(
    r'C:\Users\welld\PycharmProjects\Identification_and_classification_amazon_forest\planet\train-jpg', target_size=(150, 150), batch_size=32, subset='training', class_mode='categorical')

val_data = datagen.flow_from_directory(
    r'C:\Users\welld\PycharmProjects\Identification_and_classification_amazon_forest\planet\train-jpg', target_size=(150, 150), batch_size=32, subset='validation', class_mode='categorical')

# Load pre-trained ResNet50
resnet_base = ResNet50(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

# Add custom layers
x = Flatten()(resnet_base.output)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(len(train_data.class_indices), activation='softmax')(x)

resnet_model = Model(inputs=resnet_base.input, outputs=output)

# Freeze base model
for layer in resnet_base.layers:
    layer.trainable = False

resnet_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

resnet_model.summary()

# Train ResNet model
history_resnet = resnet_model.fit(train_data, validation_data=val_data, epochs=10)
