
#In[1]
import os
import cv2

# === CONFIGURATION ===
input_dir = r"C:\Users\Karim Sherif\Desktop\WORKSPACE01\khlas_da_will_yshtghl\Drowsiness_Detection_Dataset"
output_dir = r"C:\Users\Karim Sherif\Desktop\WORKSPACE01\khlas_da_will_yshtghl\Drowsiness_Detection_Dataset_resized"
classes = ["alert", "drowsy"]
target_size = (224, 224)

# === CREATE OUTPUT FOLDERS ===
for cls in classes:
    os.makedirs(os.path.join(output_dir, cls), exist_ok=True)

# === RESIZE AND SAVE ===
for cls in classes:
    input_path = os.path.join(input_dir, cls)
    output_path = os.path.join(output_dir, cls)

    for img_name in os.listdir(input_path):
        img_input_path = os.path.join(input_path, img_name)
        img_output_path = os.path.join(output_path, img_name)

        img = cv2.imread(img_input_path)

        if img is not None:
            resized_img = cv2.resize(img, target_size)
            cv2.imwrite(img_output_path, resized_img)
        else:
            print(f"❌ Skipped corrupted image: {img_input_path}")


# data visuals b3d 

# %%
data_dir = r"C:\Users\Karim Sherif\Desktop\WORKSPACE01\khlas_da_will_yshtghl\Drowsiness_Detection_Dataset_resized"  # e.g., "./Drowsiness_Detection_Dataset/"

classes = ["drowsy", "alert"]

shapes = set()
for cls in classes:
    path = os.path.join(data_dir, cls)
    for img_name in os.listdir(path)[:100]:  # Check first 100
        img = cv2.imread(os.path.join(path, img_name))
        if img is not None:
            shapes.add(img.shape)
        else:
            print(f"Corrupted image: {img_name}")

print("Unique image shapes:", shapes)


# %%
import os
import shutil
import random

# Paths
source_dir = "Drowsiness_Detection_Dataset_resized"
output_dir = "Drowsiness_Dataset_Split"
classes = ["alert", "drowsy"]
train_ratio = 0.8  # 80% train, 20% val

# Create output directories
for subset in ["train", "val"]:
    for cls in classes:
        os.makedirs(os.path.join(output_dir, subset, cls), exist_ok=True)

# Split data
for cls in classes:
    src_folder = os.path.join(source_dir, cls)
    all_images = os.listdir(src_folder)
    random.shuffle(all_images)

    train_cutoff = int(len(all_images) * train_ratio)
    train_images = all_images[:train_cutoff]
    val_images = all_images[train_cutoff:]

    for img_name in train_images:
        shutil.copy(
            os.path.join(src_folder, img_name),
            os.path.join(output_dir, "train", cls, img_name)
        )

    for img_name in val_images:
        shutil.copy(
            os.path.join(src_folder, img_name),
            os.path.join(output_dir, "val", cls, img_name)
        )

print("✅ Data splitting complete.")




# %%
import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import array_to_img, img_to_array, load_img
import shutil

# === CONFIG ===
input_dir = "Drowsiness_Dataset_Split"
output_dir = "Drowsiness_Dataset_Grayscale"
target_size = (224, 224)
classes = ["alert", "drowsy"]

# === STEP 1: Convert to Grayscale and Save ===
for subset in ["train", "val"]:
    for cls in classes:
        input_path = os.path.join(input_dir, subset, cls)
        output_path = os.path.join(output_dir, subset, cls)
        os.makedirs(output_path, exist_ok=True)

        for img_name in os.listdir(input_path):
            img_path = os.path.join(input_path, img_name)
            gray_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if gray_img is None:
                print(f"Skipping: {img_path}")
                continue

            resized_img = cv2.resize(gray_img, target_size)
            save_path = os.path.join(output_path, img_name)
            cv2.imwrite(save_path, resized_img)

print("✅ Grayscale conversion complete.")



# #In[]

# from tensorflow.keras.preprocessing.image import ImageDataGenerator

# # Paths to your preprocessed grayscale dataset
# train_dir = 'Drowsiness_Dataset_Grayscale/train'
# val_dir = 'Drowsiness_Dataset_Grayscale/val'

# # === STEP 2 & 3: Create ImageDataGenerators ===

# # For training — includes augmentation
# train_datagen = ImageDataGenerator(
#     rescale=1.0 / 255,
#     rotation_range=10,
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     zoom_range=0.1,
#     horizontal_flip=True,
#     fill_mode='nearest'
# )

# # For validation — only normalization
# val_datagen = ImageDataGenerator(
#     rescale=1.0 / 255
# )

# # === Load data using flow_from_directory ===

# train_generator = train_datagen.flow_from_directory(
#     train_dir,
#     target_size=(224, 224),
#     color_mode='grayscale',
#     batch_size=32,
#     class_mode='binary',
#     shuffle=True
# )

# val_generator = val_datagen.flow_from_directory(
#     val_dir,
#     target_size=(224, 224),
#     color_mode='grayscale',
#     batch_size=32,
#     class_mode='binary',
#     shuffle=False
# )

# # Check the class labels
# print("Class indices:", train_generator.class_indices)

# %%
