from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from PIL import Image

app = Flask(__name__)

model = load_model('./faceRecognition/face_model.keras')

emotion_labels = ['Happiness', 'Sadness', 'Neutral']

# Home route
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        img = Image.open(file.stream)
        img = img.convert('L')
        img = img.resize((48, 48))  
        
        img_array = np.array(img)
        img_array = img_array / 255.0 
        img_array = np.expand_dims(img_array, axis=0)  
        img_array = np.expand_dims(img_array, axis=-1)

        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions[0])
        predicted_emotion = emotion_labels[predicted_class]
        confidence = predictions[0][predicted_class]

        result_text = f"Predicted Emotion: {predicted_emotion} (Confidence: {confidence:.2f})"
        return render_template('index.html', prediction=result_text)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
