from flask import Flask, request, render_template
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
import PIL
# from keras.models import load_mode
import pickle
import os
import numpy as np
# print(pd.show_versions())


# Declare a Flask app
app = Flask(__name__)

# @app.route('/', methods=['GET', 'POST'])
# def main():

#     # If a form is submitted
#     if request.method == "POST":

#         # Unpickle classifier
#         clf = joblib.load("clf.pkl")

#         # Get values through input bars
#         height = request.form.get("height")
#         weight = request.form.get("weight")

#         # Put inputs to dataframe
#         X = pd.DataFrame([[height, weight]], columns = ["Height", "Weight"])

#         # Get prediction
#         prediction = clf.predict(X)[0]

#     else:
#         prediction = ""

#     return render_template("index.html", output = prediction)
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

def predict(model, img):
    test_img = tf.keras.preprocessing.image.load_img(img, target_size=(256, 256))
    img_arr = tf.keras.preprocessing.image.img_to_array(test_img)
    img_arr = tf.expand_dims(img_arr ,0)

    prediction = model.predict(img_arr)

    predicted_class = class_names[np.argmax(prediction[0])]
    confidence = round(100 *(np.max(prediction[0])),2)

    return predicted_class , confidence

@app.route('/', methods=['GET', 'POST'])
def imgHandler():

    # If a form is submitted
    if request.method == "POST":
        file = request.files['image']  # fet input
        filename = file.filename
        print("@@ Input posted = ", filename)
        file_path = os.path.join('static/uploaded', filename)
        file.save(file_path)

        prediction = "Image Uploaded Successfully: " + filename
        img_path = file_path
        img_path1 = 'static/uploaded/'+filename
        # # Unpickle classifier
        # model = pickle.load(open('vggmodel.pkl', 'rb'))
        # model = joblib.load("vggmodel.pkl")
        # model = joblib.load("vgg16_model.h5")
        model =load_model("vgg16_model.h5")

        predicted_class , confidence = predict(model, img_path1)
        print(predicted_class , confidence)
        # # Get values through input bars
        # height = request.form.get("height")
        # weight = request.form.get("weight")
        
        # # Put inputs to dataframe
        # X = pd.DataFrame([[height, weight]], columns = ["Height", "Weight"])
        
        # # Get prediction
        # prediction = clf.predict(X)[0]
        
    else:
        prediction = ""
        img_path = ""
        
    return render_template("index.html", output = prediction, img_path = img_path, predicted_class = predicted_class, confidence=confidence)

# Running the app
if __name__ == '__main__':
    app.run(debug = True)