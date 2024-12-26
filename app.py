import pandas as pd
import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load model TensorFlow
resnet_model = load_model('resnet_model.h5')
cnn_model = load_model('best_model.keras')

# Load class labels
csv_file_path = 'data/planets/train_classes.csv'
class_labels_df = pd.read_csv(csv_file_path)

# Remove duplicate class labels
class_labels = list(set(class_labels_df['tags'].tolist()))  # Assuming the column with class names is 'tags'
class_labels.sort()  # Optional: Sort alphabetically for consistent output

# Tentukan threshold tetap
THRESHOLD = 0.5  # 50%

# Fungsi untuk melakukan prediksi
def predict_image(image, model):
    image = image.convert('RGB')  # Pastikan gambar dalam mode RGB
    image = image.resize((224, 224))  # Sesuaikan ukuran input model
    img_array = img_to_array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    return predictions

# UI Streamlit
st.title("Amazon Forest Classification")
st.write("Aplikasi ini mengklasifikasikan citra berdasarkan model ResNet atau CNN.")

# Pilih model
model_option = st.selectbox("Pilih Model", ["ResNet", "CNN"])

# Upload gambar
uploaded_file = st.file_uploader("Upload Gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Tampilkan gambar yang di-upload
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang di-upload", use_column_width=True)

    # Prediksi
    if st.button("Klasifikasi"):
        with st.spinner("Sedang memproses..."):
            # Pilih model berdasarkan pilihan pengguna
            model = resnet_model if model_option == "ResNet" else cnn_model

            # Lakukan prediksi
            predictions = predict_image(image, model)

            # Get the top 3 predictions
            top_indices = predictions[0].argsort()[-3:][::-1]
            top_probabilities = predictions[0][top_indices]
            top_classes = [class_labels[i] for i in top_indices]

            # Filter berdasarkan threshold tetap
            filtered_predictions = [
                (class_name.capitalize(), prob * 100)
                for class_name, prob in zip(top_classes, top_probabilities)
                if prob >= THRESHOLD
            ]

            # Tampilkan hasil
            if filtered_predictions:
                st.success(f"Hasil Prediksi: {filtered_predictions[0][0]}({filtered_predictions[0][1]:.2f}%) ")
                for class_name, prob in filtered_predictions:
                    st.write(f"{class_name}: {prob:.2f}%")
            else:
                st.warning(f"Tidak ada prediksi yang melebihi threshold probabilitas ({THRESHOLD * 100:.0f}%).")

            # Show the raw prediction probabilities for transparency
            st.write("Probabilitas Lengkap:")
            for class_name, prob in zip(top_classes, top_probabilities):
                st.write(f"{class_name}: {prob * 100:.2f}%")
