import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("SpotifyAudioFeaturesApril2019.csv")

# Function to classify mood
def assign_mood(row):
    val = row['valence']
    energy = row['energy']
    dance = row['danceability']
    tempo = row['tempo']
    acoustic = row['acousticness']
    loud = row['loudness']

    if energy > 0.7 and dance > 0.7 and tempo > 115:
        return "Party"
    if energy > 0.75 and loud > -8 and (0.4 < val < 0.8):
        return "Energetic"
    if val > 0.6 and 0.4 < energy < 0.75:
        return "Happy"
    if 0.45 < val < 0.75 and dance > 0.5 and energy < 0.6:
        return "Love"
    if energy < 0.40 and acoustic > 0.4 and val > 0.4:
        return "Calm"
    if val < 0.35 and energy < 0.5:
        return "Sad"
    return "Calm"

# Apply mood classification to full dataset
df["mood"] = df.apply(assign_mood, axis=1)

# Feature selection
features = [
    'acousticness','danceability','duration_ms','energy','instrumentalness',
    'liveness','loudness','speechiness','tempo','valence'
]

X = df[features]
y = df['mood']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# Evaluate accuracy
y_pred = knn.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", round(accuracy * 100, 2), "%")

# -----------------------------------------
# 👇 ENTER NEW SONG FEATURES HERE TO PREDICT
# -----------------------------------------

# new_song = pd.DataFrame([{
#     "acousticness": 0.20,
#     "danceability": 0.65,
#     "duration_ms": 210000,
#     "energy": 0.70,
#     "instrumentalness": 0.05,
#     "liveness": 0.10,
#     "loudness": -6.0,
#     "speechiness": 0.06,
#     "tempo": 125.0,
#     "valence": 0.75
# }])
new_song = pd.DataFrame([{
    "acousticness": 0.72,
    "danceability": 0.28,
    "duration_ms": 210000,   # 3 min 30 sec
    "energy": 0.22,
    "instrumentalness": 0.05,
    "liveness": 0.11,
    "loudness": -13.5,
    "speechiness": 0.03,
    "tempo": 82.0,
    "valence": 0.18
}])

new_scaled = scaler.transform(new_song)
predicted_mood = knn.predict(new_scaled)

print("\nPredicted Mood of the song:", predicted_mood[0])
