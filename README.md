# 🎥 IMDb Movie Rating Predictor

Welcome to the **IMDb Movie Rating Predictor** — an intelligent Streamlit web application that predicts the IMDb rating of a movie based on its **actors, director, genres**, and **plot summary** using advanced machine learning models like **XGBoost** and **NLP Sentence Transformers**.

---

## 📸 App Preview

Here's a screenshot of the app in action:

![image](https://github.com/user-attachments/assets/2010cc81-475e-4696-ab11-0d46a01f3d5a)

---

## 💡 Features

- 🎭 Analyze actor and director impact using their average ratings
- 🧠 Use Sentence-BERT to convert movie plot summaries into numerical features
- 🎬 Handle multi-genre inputs via one-hot encoding
- ⚙️ Make real-time predictions using a trained XGBoost model
- 🖼️ Clean and interactive Streamlit UI
- ⚡ Fast, intuitive, and responsive performance

---

## 🧠 How It Works

1. **User Inputs**:
   - Actors (comma-separated)
   - Director name
   - Genres (comma-separated)
   - Movie synopsis (short summary)

2. **Feature Engineering**:
   - Average IMDb ratings for actors and directors (from training data)
   - Genre one-hot encoding using stored columns
   - Sentence embedding using `sentence-transformers` (BERT-based)

3. **Model Prediction**:
   - Features are passed to a trained **XGBoost Regressor**
   - Final IMDb rating is predicted and shown

---

## 🌟 Future Improvements
 -🌍 Deploy the app on Hugging Face / Render / Heroku

 -🔍 Add autocomplete for actors and directors

 -🎯 Show model confidence or prediction intervals

 -🎥 Recommend similar movies based on synopsis

## 🙋‍♀️ About the Author
Developed by Richa Shukla, an aspiring Data Scientist passionate about NLP, MLOps, and real-world AI applications. 

## 📜 License
This project is licensed under the MIT License.
