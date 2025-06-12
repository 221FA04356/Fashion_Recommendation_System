import os
import numpy as np
import pickle
from tqdm import tqdm
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model

IMG_DIR = "images"  # Folder with your 44,441 images
SIZE = (224, 224)

# Load MobileNetV2 model with pretrained weights
base_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
model = Model(inputs=base_model.input, outputs=base_model.output)

def extract_features():
    features = []
    image_names = []

    for img_name in tqdm(os.listdir(IMG_DIR)):
        img_path = os.path.join(IMG_DIR, img_name)
        try:
            img = image.load_img(img_path, target_size=SIZE)
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)

            feature = model.predict(img_array, verbose=0)[0]
            features.append(feature)
            image_names.append(img_name)
        except Exception as e:
            print(f"Skipped {img_name}: {e}")

    features = np.array(features)
    with open("model.pkl", "wb") as f:
        pickle.dump((features, image_names), f)

if __name__ == "__main__":
    extract_features()
