import pandas as pd

spotify_df = pd.read_csv("SpotifyData/data_moods.csv")
x = "Happy"
df = []
if x == "Disgust":
    df = spotify_df[spotify_df['mood'].isin(['Energetic', 'Happy', 'Calm'])]
if x == "Angry":
    df = spotify_df[spotify_df['mood'].isin(['Calm'])]
if x == "Fear":
    df = spotify_df[spotify_df['mood'].isin(['Happy', 'Calm'])]
if x == "Happy":
    df = spotify_df[spotify_df['mood'].isin(['Sad', 'Happy', 'Calm'])]
if x == "Sad":
    df = spotify_df[spotify_df['mood'].isin(['Energetic', 'Happy', 'Calm'])]
if x == "Surprise":
    df = spotify_df[spotify_df['mood'].isin(['Energetic', 'Happy', 'Sad'])]

df = df.sample(n=10)
name_list = df["name"].tolist()
artist_list = df["artist"].tolist()
final_dict = dict(zip(name_list, artist_list))

print(final_dict)
