import streamlit as st
import pandas as pd
import joblib

# Load assets
model = joblib.load("knn_music_model.pkl")
scaler = joblib.load("scaler.pkl")

df = pd.read_csv("SpotifyAudioFeaturesApril2019.csv")

st.title("🎵 Music Mood Prediction using KNN")

st.write("Enter the features of the song:")

acousticness = st.slider("Acousticness", 0.0, 1.0, 0.5)
danceability = st.slider("Danceability", 0.0, 1.0, 0.5)
duration_ms = st.number_input("Duration (ms)", min_value=30000, max_value=600000, value=210000)
energy = st.slider("Energy", 0.0, 1.0, 0.5)
instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.0)
liveness = st.slider("Liveness", 0.0, 1.0, 0.3)
loudness = st.slider("Loudness", -60.0, 0.0, -10.0)
speechiness = st.slider("Speechiness", 0.0, 1.0, 0.1)
tempo = st.slider("Tempo", 40.0, 220.0, 120.0)
valence = st.slider("Valence", 0.0, 1.0, 0.5)

if st.button("Predict Mood"):
    new_song = pd.DataFrame([{
        "acousticness": acousticness,
        "danceability": danceability,
        "duration_ms": duration_ms,
        "energy": energy,
        "instrumentalness": instrumentalness,
        "liveness": liveness,
        "loudness": loudness,
        "speechiness": speechiness,
        "tempo": tempo,
        "valence": valence
    }])

    new_scaled = scaler.transform(new_song)
    mood = model.predict(new_scaled)[0]

    st.success(f"🎧 **Predicted Mood: {mood}**")
