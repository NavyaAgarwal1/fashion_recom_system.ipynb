import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

# Load feature list and filenames
try:
    feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
    filenames = pickle.load(open('filenames.pkl', 'rb'))
except Exception as e:
    st.error(f"Error loading pickle files: {e}")
    st.stop()

# Load pre-trained ResNet50 model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
model.trainable = False
model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

st.title('Fashion Recommender System')

def save_uploaded_file(uploaded_file):
    try:
        upload_folder = 'uploads'
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)  # Create the 'uploads' folder if it doesn't exist
        with open(os.path.join(upload_folder, uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return 0

def feature_extraction(img_path, model):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        expanded_img_array = np.expand_dims(img_array, axis=0)
        preprocessed_img = preprocess_input(expanded_img_array)
        result = model.predict(preprocessed_img).flatten()
        normalized_result = result / norm(result)
        return normalized_result
    except Exception as e:
        st.error(f"Error processing {img_path}: {e}")
        return None

def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices

# File upload
uploaded_file = st.file_uploader("Choose an image")
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        # Display the uploaded image
        display_image = Image.open(uploaded_file)
        st.image(display_image)

        # Feature extraction
        features = feature_extraction(os.path.join("uploads", uploaded_file.name), model)
        if features is not None:
            # Recommendation
            indices = recommend(features, feature_list)

            # Display recommended images
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.image(filenames[indices[0][0]])
            with col2:
                st.image(filenames[indices[0][1]])
            with col3:
                st.image(filenames[indices[0][2]])
            with col4:
                st.image(filenames[indices[0][3]])
            with col5:
                st.image(filenames[indices[0][4]])
        else:
            st.error("Feature extraction failed!")
    else:
        st.error("Some error occurred in file upload.")
