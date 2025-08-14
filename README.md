# Movie Recommender System 
This project implements a movie recommendation system leveraging a Softmax Deep Neural Network (DNN) trained on the MovieLens 100K dataset. It provides personalized movie recommendations based on user interaction history and offers an interactive web dashboard built with Dash to visualize the recommendations.

---

## Overview
- **Dataset:** MovieLens 100K (943 users, 1682 movies, 100K ratings)

- **Model:** Softmax Deep Neural Network with user/movie embeddings

- **Web Interface:** Built using Dash for real-time recommendations

- **Goal:** Generate personalized movie suggestions using deep learning

---

## Technologies Used
- Python 

- TensorFlow / Keras

- Pandas, NumPy

- Plotly Dash 

- scikit-learn

- Matplotlib

---

## Model Architecture
**Neural Network (Softmax DNN)**
- **Inputs:** User IDs & Movie IDs (encoded)

- **Embeddings:** 150-dim vectors for users and movies

- **Hidden Layers:** Two Dense layers (32 → 16 neurons) with ReLU

- **Regularization:** Dropout (0.05), L2 on embeddings

- **Output:** 9-class Softmax predicting rating probabilities (1–5 scale)

- **Optimizer:** SGD

- **Loss:** SparseCategoricalCrossentropy

- **Epochs:** 70, Batch Size: 128

---

## Recommendation Logic
Predicts unseen movies for a user

Scores them using max Softmax probability

Ranks & recommends top-N movies

Dash app displays recommendations, seen movies & genre chart

---

## Web Dashboard (Dash)
Input a User ID

View movies already seen

Explore recommended movies

Visualize genre distribution

---

## Project Structure

```text
├── ml-100k/                      # Dataset files  
├── app.py                        # Dash web app  
├── movie_recommender_model.h5    # Trained model  
├── Recommender-System-*.ipynb    # Training notebook  
├── requirements.txt              # Dependencies  
└── README.md

---
