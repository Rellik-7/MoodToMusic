from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import cv2
import numpy as np




app = Flask(__name__)

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}


model = load_model('model_val67/model_fer2013_val67.h5')

model.make_predict_function()

spotify_df = read_csv("SpotifyData/data_moods.csv")
def ChooseDataset(x):
    if x == "Disgust":
        df_1 = spotify_df[spotify_df['Mood'].isin(['Energetic', 'Happy', 'Calm'])]
    if x == "Angry":
        df_1 = spotify_df[spotify_df['Mood'].isin(['Calm'])]
    if x == "Fear":
        df_1 = spotify_df[spotify_df['Mood'].isin(['Happy', 'Calm'])]
    if x == "Happy":
        df_1 =  spotify_df[spotify_df['Mood'].isin(['Sad', 'Happy', 'Calm'])]
    if x == "Sad":
        df_1 =  spotify_df[spotify_df['Mood'].isin(['Energetic','Happy'])]
    if x == "Surprise":
        df_1 =  spotify_df[spotify_df['Mood'].isin(['Energetic','Happy','Sad'])]
	
    df = df_1.sample(n=10)
    dict = {df["name"]:df["artist"]}
    return dict

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
	cv2.imwrite(img_path,frame)
	return results[0]


# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

# @app.route("/about")
# def about_page():
# 	return "Please subscribe  Artificial Intelligence Hub..!!!"

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/" + img.filename
		img.save(img_path)

		p = predict_label(img_path)
		your_dict = chooseDataset(p)

	return render_template("index.html", prediction = p, img_path = img_path, your_dict = your_dict)






if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)


	

