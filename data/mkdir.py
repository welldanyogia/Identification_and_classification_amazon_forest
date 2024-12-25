import os
import shutil
import pandas as pd

# Membaca file CSV yang berisi informasi kelas
df = pd.read_csv('planets/test_classes.csv')

# Path ke direktori gambar
train_dir = 'planets/test-jpg'

# Membuat subfolder sesuai kelas yang ada
for _, row in df.iterrows():
    class_name = row['tags']  # Gantilah 'tags' dengan nama kolom yang benar
    image_name = row['image_name'] + '.jpg'  # Menambahkan ekstensi .jpg

    # Membuat folder kelas jika belum ada
    class_folder = os.path.join(train_dir, class_name)
    if not os.path.exists(class_folder):
        os.makedirs(class_folder)

    # Memastikan file sumber ada sebelum memindahkan
    src_image_path = os.path.join(train_dir, image_name)
    if os.path.exists(src_image_path):
        dest_image_path = os.path.join(class_folder, image_name)
        shutil.move(src_image_path, dest_image_path)
        print(f"Memindahkan {image_name} ke {class_name}")
    else:
        print(f"File {image_name} tidak ditemukan di {src_image_path}")

print("Gambar berhasil dipindahkan ke subfolder sesuai kelas.")