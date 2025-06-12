import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle

with open("model.pkl", "rb") as f:
    features, image_names = pickle.load(f)

def recommend(uploaded_feature, top_k=5):
    similarities = cosine_similarity([uploaded_feature], features)[0]
    indices = np.argsort(similarities)[-top_k:][::-1]
    return [image_names[i] for i in indices]
