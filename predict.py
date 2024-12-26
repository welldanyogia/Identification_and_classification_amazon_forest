import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model

# Load your trained model
# loaded_resnet_model = load_model('best_model.keras')
loaded_resnet_model = tf.keras.models.load_model('best_model.keras')

# Load class labels from the CSV file
csv_file_path = 'data/planets/train_classes.csv'
class_labels_df = pd.read_csv(csv_file_path)

# Remove duplicate class labels
class_labels = list(set(class_labels_df['tags'].tolist()))  # Assuming the column with class names is 'tags'
class_labels.sort()  # Optional: Sort alphabetically for consistent output

# Path to the new image
image_path = 'planet/test-jpg/test_1000.jpg'

# Preprocess the image
# img = load_img(image_path, target_size=(224, 224))  # Adjust to the model's input shape
# img_array = img_to_array(img) / 255.0  # Normalize as done during training
# img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

img = Image.open(image_path)
img = img.convert('RGB')
img = img.resize((224, 224))
img_array = np.array(img)
img_array = np.expand_dims(img_array, axis=0)

# Make predictions using the model
predictions = loaded_resnet_model.predict(img_array)
predicted_class_index = np.argmax(predictions)  # Get the class index with the highest probability
predicted_class_name = class_labels[predicted_class_index]  # Map index to class name

# Print the predicted class
print(f"Predicted class: {predicted_class_name} (Index: {predicted_class_index})")
# print(f"Prediction probabilities: {predictions[0]}")
# print(f"Prediction : {predictions}")
#
# # Print all class tags with their indices
# print("\nClass Tags with Indices (No Duplicates):")
# for index, tag in enumerate(class_labels):
#     print(f"Index {index}: {tag}")

# import numpy as np
# import pandas as pd
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
# from tensorflow.keras.models import load_model
#
# # Load your trained model
# loaded_resnet_model = load_model('best_model.keras')
#
# # Load class labels from the CSV file
# csv_file_path = 'data/planets/train_classes.csv'
# class_labels_df = pd.read_csv(csv_file_path)
# class_labels = class_labels_df['tags'].tolist()  # Assuming the column with class names is 'class'
#
# # Path to the new image
# image_path = 'data/planets/test-jpg/water/test_24448.jpg'
#
# # Preprocess the image
# img = load_img(image_path, target_size=(224, 224))  # Adjust to the model's input shape
# img_array = img_to_array(img) / 255.0  # Normalize as done during training
# img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
#
# # Make predictions using the model
# predictions = loaded_resnet_model.predict(img_array)
# predicted_class_index = np.argmax(predictions)  # Get the class index with the highest probability
# predicted_class_name = class_labels[predicted_class_index]  # Map index to class name

# Print the results
# print(f"Predicted class: {predicted_class_name} (Index: {predicted_class_index}), Probabilities: {predictions}, Class Tags : {class_labels}")
# Print the predicted class
# print(f"Predicted class: {predicted_class_name} (Index: {predicted_class_index})")
# print(f"Prediction probabilities: {predictions[0]}")
#
# # Print all class tags with their indices
# print("\nClass Tags with Indices:")
# for index, tag in enumerate(class_labels):
#     print(f"Index {index}: {tag}")

# import numpy as np
# import pandas as pd
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
#
# # Load your trained model
# loaded_resnet_model = load_model('best_model.keras')
#
# # Load class labels from the CSV file
# csv_file_path = 'data/planets/train_classes.csv'
# class_labels_df = pd.read_csv(csv_file_path)
# class_labels = class_labels_df['tags'].tolist()  # Assuming the column with class names is 'class'
#
# # Set up ImageDataGenerator for test data preprocessing
# test_datagen = ImageDataGenerator(rescale=1./255)  # Rescale pixel values (same as during training)
# test_dir = 'data/planets/test-jpg'  # The directory where test images are stored
#
# # Prepare test data generator (assuming you have a subfolder for each class)
# test_generator = test_datagen.flow_from_directory(
#     test_dir,
#     target_size=(224, 224),  # Resize to model input size
#     batch_size=32,  # Use appropriate batch size
#     class_mode='categorical',  # Since we are doing multi-class classification
#     shuffle=False  # Important: do not shuffle test data during evaluation
# )
#
# # Evaluate the model on the test dataset
# test_loss, test_accuracy = loaded_resnet_model.evaluate(test_generator, verbose=1)
#
# # Print evaluation results
# print(f"Test Loss: {test_loss}")
# print(f"Test Accuracy: {test_accuracy}")
#
# # If you want to print detailed predictions for each image:
# predictions = loaded_resnet_model.predict(test_generator, verbose=1)
# predicted_class_indices = np.argmax(predictions, axis=1)  # Get class index with highest probability
#
# # Map predicted indices to class names
# predicted_class_names = [class_labels[i] for i in predicted_class_indices]
#
# # Print some of the predictions (example: first 5 predictions)
# for i in range(5):
#     print(f"Image {i+1}: Predicted class: {predicted_class_names[i]} (Index: {predicted_class_indices[i]}), Probability: {predictions[i]}")
