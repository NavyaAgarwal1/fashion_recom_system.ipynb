import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import numpy as np
from numpy.linalg import norm
import os
from tqdm import tqdm
import pickle

# Load the model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
model.trainable = False
model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

def extract_features(img_path, model):
    try:
        img = image.load_img(img_path, target_size=(224,224))
        img_array = image.img_to_array(img)
        expanded_img_array = np.expand_dims(img_array, axis=0)
        preprocessed_img = preprocess_input(expanded_img_array)
        result = model.predict(preprocessed_img).flatten()
        normalized_result = result / norm(result)
        return normalized_result
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None

# Directory containing images
image_dir = '/Users/nav/Downloads/archive/myntradataset/images'
filenames = [
    os.path.join(image_dir, file) 
    for file in os.listdir(image_dir) 
    if file.lower().endswith(('.png', '.jpg', '.jpeg'))
]

# Extract and save features
feature_list = []
for file in tqdm(filenames, desc="Extracting features"):
    features = extract_features(file, model)
    if features is not None:
        feature_list.append(features)

# Save features and filenames
with open('embeddings.pkl', 'wb') as f:
    pickle.dump(feature_list, f)
with open('filenames.pkl', 'wb') as f:
    pickle.dump(filenames, f)

print("Feature extraction and saving completed.")