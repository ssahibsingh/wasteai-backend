import numpy as np
import tensorflow as tf
import gc
import imghdr
from bson.binary import Binary
from flask import Flask, request, jsonify, has_request_context
from flask_cors import CORS, cross_origin
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as image_utils
from io import BytesIO

# from memory_profiler import profile


app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})


# Get and Post Request handler
@app.route("/", methods=["GET", "POST"])
@cross_origin()
# @profile
def home():
    if request.method == "GET":
        gc.collect()
        return jsonify({"status": "active", "message": "Waste AI"}), 200
    elif request.method == "POST":
        image = request.files["image"]
        image_data = Binary(image.read())
        image_format = imghdr.what(None, h=image_data)

        model = load_model("models/vgg16_model.h5")
        # Model Classes
        class_names = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

        # print("**********\n Predicting.... \n\n")
        test_img = image_utils.load_img(BytesIO(image_data), target_size=(256, 256))
        img_arr = image_utils.img_to_array(test_img)
        img_arr = tf.expand_dims(img_arr, 0)

        prediction = model.predict(img_arr)

        predicted_class = class_names[np.argmax(prediction[0])]
        confidence = round(100 * (np.max(prediction[0])), 2)
        # print("Predicted Class: ",predicted_class)
        # print("Confidence: ",confidence)
        # print("**********\n")

        del test_img, img_arr, prediction, model, image, image_data, image_format
        gc.collect()

        return jsonify(
            {
                "success": "true",
                "message": "Prediction Successful",
                "prediction": predicted_class,
                "confidence": confidence,
                # 'image_id': str(image_id),
            }
        )

# Running the app
if __name__ == "__main__":
    app.run(debug=True)
