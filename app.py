import pandas as pd
import numpy as np
import PIL
import pickle
import os
import tensorflow as tf
import joblib
from bson.binary import Binary
from flask import Flask, request, render_template, jsonify, send_file
from flask_pymongo import pymongo
from tensorflow.keras.models import load_model
from dotenv import load_dotenv
import imghdr
from io import BytesIO
from bson.objectid import ObjectId


load_dotenv()
# Declare a Flask app
app = Flask(__name__)

# MongoDB Connection
client = pymongo.MongoClient(os.getenv('MONGODB_URI'))
db = client.get_database('uploaded_image')

# Model Classes
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']


def predict(model, img):
    test_img = tf.keras.preprocessing.image.load_img(img, target_size=(256, 256))
    img_arr = tf.keras.preprocessing.image.img_to_array(test_img)
    img_arr = tf.expand_dims(img_arr ,0)

    prediction = model.predict(img_arr)

    predicted_class = class_names[np.argmax(prediction[0])]
    confidence = round(100 *(np.max(prediction[0])),2)

    return predicted_class , confidence

@app.route('/image/<image_id>', methods=['GET'])
def get_image(image_id):
    image = db.images.find_one({"_id": ObjectId(image_id)})
    if image is None:
        return jsonify({'status': 'error', 'message': 'Image not found'}), 404
    else:
        image_data = image['image']
        image_format = image['format']
        return send_file(BytesIO(image_data), mimetype=('image/'+image_format))

# @app.route('/')
# def home():
#     prediction = ""
#     img_path = ""
#     predicted_class=""
#     confidence=""
#     return render_template('index.html')


@app.route('/', methods=['GET','POST'])
def home():
    if request.method == 'GET':
        prediction = ""
        image_id = ""
        predicted_class=""
        confidence=""
        return render_template('index.html', output = prediction, image_id = image_id, predicted_class = predicted_class, confidence = confidence)
    else:
        image = request.files['image']  # fet input
        image_data = Binary(image.read())
        image_format = imghdr.what(None, h=image_data)
        alreadyInDB = db.images.find_one({"image": image_data})  # check if image already exists in database
        if alreadyInDB is None:
            image_id = db.images.insert_one({"image": image_data, "format":image_format}).inserted_id
        else:
            image_id = alreadyInDB['_id']


        # print(image_id)
        # print("\n")
        # print("*****Binary Image: \n*****")
        # print(Binary(image_data) == image_data)
        # filename = file.filename
        # print("@@ Input posted = ", filename)
        # file_path = os.path.join('static/uploaded', filename)
        # file.save(file_path)

        prediction = "Image Uploaded Successfully"
        # img_path = file_path
        # img_path1 = 'static/uploaded/'+filename
        # # # Unpickle classifier
        # # model = pickle.load(open('vggmodel.pkl', 'rb'))
        # # model = joblib.load("vggmodel.pkl")
        # # model = joblib.load("vgg16_model.h5")
        model =load_model("vgg16_model.h5")
        # img_path = BytesIO(image_data)
        predicted_class , confidence = predict(model, BytesIO(image_data))
        # print(predicted_class , confidence)
        # Get values through input bars
        # height = request.form.get("height")
        # weight = request.form.get("weight")
        
        # # Put inputs to dataframe
        # X = pd.DataFrame([[height, weight]], columns = ["Height", "Weight"])
        
        # # Get prediction
        # prediction = clf.predict(X)[0]
        # prediction = ""
        # img_response = ""
        # predicted_class=""
        # confidence=""
        return render_template("index.html", output = prediction, image_id = image_id, predicted_class = predicted_class, confidence=confidence)
        # return jsonify({'status': 'success', 'image_id': str(image_id)})

# Running the app
if __name__ == '__main__':
    app.run(debug = True)