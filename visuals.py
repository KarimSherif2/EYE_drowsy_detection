#In[1]
import matplotlib.pyplot as plt
import os

#In[2]
data_dir = r"C:\Users\Karim Sherif\Desktop\WORKSPACE01\khlas_da_will_yshtghl\Drowsiness_Detection_Dataset"  # e.g., "./Drowsiness_Detection_Dataset/"
classes = ["drowsy", "alert"]

for cls in classes:
    cls_path = os.path.join(data_dir, cls)
    num_images = len(os.listdir(cls_path))
    print(f"{cls} ({'drowsy' if 'Closed' in cls else 'alert'}): {num_images} images")


#In[3]

class_names = ["drowsy", "alert"]
class_counts = [len(os.listdir(os.path.join(data_dir, cls))) for cls in classes]

#In[4]

plt.bar(class_names, class_counts, color=['red', 'green'])
plt.title("Class Distribution")
plt.xlabel("Class")
plt.ylabel("Number of Images")
plt.show()

# %%
import cv2
import random

for cls in classes:
    path = os.path.join(data_dir, cls)
    sample_images = random.sample(os.listdir(path), 5)

    for img_name in sample_images:
        img_path = os.path.join(path, img_name)
        img = cv2.imread(img_path)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(f"{cls} → {'drowsy' if 'Closed' in cls else 'alert'}")
        plt.axis('off')
        plt.show()



# %%
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
