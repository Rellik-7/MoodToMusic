from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import cv2
import numpy as np
import pandas as pd

app = Flask(__name__)

emotion_dict = {0: "Angry", 1: "Disgust", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

model = load_model('model_val67/model_fer2013_val67.h5')


# def recommend_songs(x):
#     spotify_df = pd.read_csv("SpotifyData/data_moods.csv")
#     df = []
#     if x == "Disgust":
#         df = spotify_df[spotify_df['mood'].isin(['Energetic', 'Happy', 'Calm'])]
#     if x == "Angry":
#         df = spotify_df[spotify_df['mood'].isin(['Calm'])]
#     if x == "Fear":
#         df = spotify_df[spotify_df['mood'].isin(['Happy', 'Calm'])]
#     if x == "Happy":
#         df = spotify_df[spotify_df['mood'].isin(['Sad', 'Happy', 'Calm'])]
#     if x == "Sad":
#         df = spotify_df[spotify_df['mood'].isin(['Energetic', 'Happy', 'Calm'])]
#     if x == "Surprise":
#         df = spotify_df[spotify_df['mood'].isin(['Energetic', 'Happy', 'Sad'])]
#
#     df = df.sample(n=10)
#     name_list = df["name"].tolist()
#     artist_list = df["artist"].tolist()
#     final_dict = dict(zip(name_list, artist_list))
#     return final_dict


model.make_predict_function()


def predict_label(img_path):
    frame = cv2.imread(img_path)
    face_detector = cv2.CascadeClassifier('haar_cascade/haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
    results = []
    bounded_frame = []
    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (0, 255, 0), 4)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        # cv2.imshow('Grayface', roi_gray_frame)
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
        cropped_img = cropped_img.astype("float") / 255.0
        emotion_prediction = model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        results.append(emotion_dict[maxindex])

    # i = image.img_to_array(i)/255.0
    # i = i.reshape(1, 100,100,3)
    # p = model.predict_classes(i)
    cv2.imwrite(img_path, frame)
    spotify_df = pd.read_csv("SpotifyData/data_moods.csv")
    df = []
    x = results[0]
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
    id_list = df["id"].tolist()
    final_dict = zip(name_list,artist_list,id_list)
    return results[0], final_dict


# routes
@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("index.html")


# @app.route("/about")
# def about_page():
# 	return "Hello!"

@app.route("/submit", methods=['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        img = request.files['my_image']

        img_path = "static/" + img.filename
        img.save(img_path)
        final_dict = {"key":"value"}
        p, final_dict = predict_label(img_path)

    return render_template("index.html", prediction=p, img_path=img_path, final_dict=final_dict)


if __name__ == '__main__':
    # app.debug = True
    app.run(debug=True)
