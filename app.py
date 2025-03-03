import os
import numpy as np
import tensorflow as tf
import cv2
from flask import Flask, render_template, request, redirect
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load trained model
MODEL_PATH = "model.h5"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model file not found. Train and save 'model.h5' first.")

model = tf.keras.models.load_model(MODEL_PATH)

# Define class names
class_names = ['butterfly', 'cat', 'chicken', 'cow', 'dog', 'elephant',
               'horse', 'sheep', 'spider', 'squirrel']


# Image preprocessing function
def preprocess_image(img_path):
    IMG_SIZE = (64, 64)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img


@app.route("/", methods=["GET", "POST"])
def index():
    filename = None
    prediction = None
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)

        file = request.files["file"]

        if file.filename == "":
            return redirect(request.url)

        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)

            # Preprocess image and make a prediction
            img_array = preprocess_image(file_path)
            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions[0])
            predicted_class_name = class_names[predicted_class]

            prediction = predicted_class_name

    return render_template("index.html", filename=filename, prediction=prediction)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80, debug=True)
