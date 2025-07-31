import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input

# Model ve sınıf ayarları
MODEL_PATH = "model/best_model.keras"
IMAGE_FOLDER = "predict_images"
class_names = ['cloudy', 'rain', 'shine', 'snow', 'storm-fog', 'sunrise']

print("Model yükleniyor...")
model = load_model(MODEL_PATH)
print("Model yüklendi.")


# Görselleri bul
image_paths = []
for root, dirs, files in os.walk(IMAGE_FOLDER):
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_paths.append(os.path.join(root, file))

# Tüm görselleri ve tahminleri sakla
images = []
titles = []

for img_path in image_paths:
    img = image.load_img(img_path, target_size=(224, 224))
    # Görseli yükle
    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array)  # <-- Bu satır kritik
    img_array = np.expand_dims(img_array, axis=0)

    # Tahmin
    preds = model.predict(img_array, verbose=0)
    pred_idx = np.argmax(preds[0])
    pred_label = class_names[pred_idx]
    confidence = preds[0][pred_idx]

    images.append(img)
    titles.append(f"{pred_label}\n({confidence:.2f}%)")


# Görselleştir
cols = 4
rows = int(np.ceil(len(images) / cols))
plt.figure(figsize=(15, 9))

for i in range(len(images)):
    plt.subplot(rows, cols, i + 1)
    plt.imshow(images[i])
    plt.title(titles[i], fontsize=10)
    plt.axis("off")

plt.savefig('results.jpg')
plt.tight_layout()
plt.show()

