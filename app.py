from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import os
from PIL import Image
import tensorflow as tf
import numpy as np

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

app = Flask(__name__, template_folder='templates')

# Set the upload folder and allowed file extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

import keras

def preprocess_image(uploaded_image_path, target_size=(224, 224)):
    img = keras.preprocessing.image.load_img(uploaded_image_path, target_size=target_size)
    img = img.convert('RGB')
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    # img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)

    return img_array

# Route to serve the HTML page (index.html)
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle image upload
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        return jsonify({'result': 'No image part'}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({'result': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # # Preprocess the image using TensorFlow functions
        try:
            image = preprocess_image(filepath)
        except Exception as e:
            return jsonify({'result': f'Error preprocessing image: {str(e)}'}), 500

        # Load your pre-trained TensorFlow model (make sure the model is saved in TensorFlow format)
        model = keras.models.load_model('models\my_model.keras')  # Replace with your model path
        
        # Perform prediction
        predictions = model.predict(image)
        
        # Get the class with the highest probability (if classification)
        predicted_class = (predictions > 0.5).astype("int32")

        class_labels = ['NORMAL', 'PNEUMONIA']
        
        predicted_class= class_labels[int(predicted_class)]
        
        # Example result (replace with actual prediction logic)
        result = f'{predicted_class}'

        return jsonify({'result': result})

    return jsonify({'result': 'Invalid file type. Only images are allowed.'}), 400

if __name__ == '__main__':
    app.run(debug=True)