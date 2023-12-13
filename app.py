from flask import Flask, render_template, request
import tensorflow as tf
from keras.models import load_model
import numpy as np
from PIL import Image
from keras.preprocessing import image

app = Flask(__name__)

# Load the pre-trained Keras model
model = load_model('Batik_mobilenet.h5')

# Function to preprocess an image for the model
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the uploaded file from the request
        file = request.files['file']

        # Save the file to the server
        file.save('static/uploaded_image.jpg')

        # Preprocess the image
        img_path = 'static/uploaded_image.jpg'
        processed_image = preprocess_image(img_path)

        # Make prediction
        prediction = model.predict(processed_image)

        # You may need to post-process the prediction based on your model output
        # For example, if it's a binary classification, you might round the prediction
        # For multiclass, you might find the class with the highest probability

        result = f"Predicted class: {prediction}"

        return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
