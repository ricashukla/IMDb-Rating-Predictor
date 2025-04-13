import numpy as np
import pickle
import streamlit as st

# Load Models and Assets
xgb_model = pickle.load(open("model.pkl", "rb"))
sentence_model = pickle.load(open("sentence_model.pkl", "rb"))
genres_columns = pickle.load(open("genre_columns.pkl", "rb"))
avg_actor_rating = pickle.load(open("avg_actor_rating.pkl", "rb"))
avg_director_rating = pickle.load(open("avg_director_rating.pkl", "rb"))

# App Title & Instructions
st.set_page_config(page_title="IMDb Rating Predictor", page_icon="🎬")
st.title("🎬 Movie Rating Predictor")
st.markdown("Provide the movie's details below and get an estimated **IMDb rating** instantly!")

# Inputs
actors = st.text_input("👥 Enter Actors")
directors = st.text_input("🎬 Enter Director")
genres = st.text_input("🏷️ Enter Genres")
synopsis = st.text_area("📝 Enter Movie Synopsis")

# Predict Button
if st.button("📊 Predict Rating"):
    with st.spinner("🔍 Analyzing and Predicting..."):
        try:
            # Handle actors/directors missing values
            input_actor_avg = np.nanmean([
                avg_actor_rating.get(actor.strip(), np.nan)
                for actor in actors.split(",") if actor.strip()
            ])

            input_director_avg = np.nanmean([
                avg_director_rating.get(director.strip(), np.nan)
                for director in directors.split(",") if director.strip()
            ])

            # Fallback to overall mean if NaN
            overall_mean = np.nanmean(list(avg_actor_rating.values()) + list(avg_director_rating.values()))
            input_actor_avg = input_actor_avg if not np.isnan(input_actor_avg) else overall_mean
            input_director_avg = input_director_avg if not np.isnan(input_director_avg) else overall_mean

            # Synopsis Embedding
            synopsis_embedding = sentence_model.encode([synopsis])

            # Genre Encoding
            input_genres = [g.strip() for g in genres.split(",") if g.strip()]
            genres_vector = np.zeros((1, len(genres_columns)))
            for genre in input_genres:
                if genre in genres_columns:
                    genres_vector[0, genres_columns.index(genre)] = 1

            # Final Feature Vector
            features = np.hstack((synopsis_embedding, genres_vector, [[input_actor_avg, input_director_avg]]))

            # Predict
            predicted_rating = xgb_model.predict(features)[0]
            st.success(f"⭐ Predicted IMDb Rating: **{predicted_rating:.2f} / 10**")

        except Exception as e:
            st.error(f"❌ Error during prediction: {str(e)}")
