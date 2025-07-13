import numpy as np
import cv2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model


MODEL_PATH = "model/model.h5"       
IMAGE_PATH = "predict_images/cardboard.jpg"        
OUTPUT_PATH = "prediction_output.jpg"

class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic']

print("Model yükleniyor...")
model = load_model(MODEL_PATH)
print("Model yüklendi.")

img = image.load_img(IMAGE_PATH, target_size=(224, 224))
img_array = image.img_to_array(img) # PIL -> np.array 
img_array = img_array / 255.0 # normalization
img_array = np.expand_dims(img_array, axis=0)  # (224, 224, 3) -> (1, 224, 224, 3) for batch

original_img = cv2.imread(IMAGE_PATH)
if original_img is None:
    raise FileNotFoundError(f"Görsel bulunamadı: {IMAGE_PATH}")
original_img = cv2.resize(original_img, (224, 224))

predictions = model.predict(img_array)
predicted_index = np.argmax(predictions[0])
predicted_label = class_names[predicted_index]
confidence = predictions[0][predicted_index] * 100

label_text = f"{predicted_label} ({confidence:.2f}%)"
cv2.putText(original_img, label_text, (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

cv2.imwrite(OUTPUT_PATH, original_img)
print(f"Tahmin: {label_text}")
print(f"Tahmin görseli kaydedildi: {OUTPUT_PATH}")
