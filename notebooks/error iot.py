from flask import Flask, Response, stream_with_context, request, jsonify
from concurrent.futures import ThreadPoolExecutor
import time
import cv2
import os
import signal
import sys
import numpy as np
from tensorflow.keras.models import load_model
from io import BytesIO
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from flask import render_template_string

app = Flask(__name__)

# Load the pre-trained model
model_path = 'C:/Users/Akoba/Desktop/START up/Covid19_research_project/models/cnn/imagemodel_covid_classif.h5'

try:
    model = load_model(model_path)
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

# Initialize some global variables to store metrics
metrics = {
    'accuracy': 0,
    'precision': 0,
    'recall': 0,
    'f1-score': 0,
    'confusion_matrix': None
}

@app.route('/metrics')
def metrics_page():
    global metrics
    return render_template_string('''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta http-equiv="refresh" content="5">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Model Metrics</title>
        <style>
            body { font-family: Arial, sans-serif; }
            table { width: 100%; border-collapse: collapse; }
            table, th, td { border: 1px solid black; }
            th, td { padding: 10px; text-align: center; }
        </style>
    </head>
    <body>
        <h2>Real-Time Model Evaluation Metrics</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Accuracy</td><td>{{ accuracy }}</td></tr>
            <tr><td>Precision</td><td>{{ precision }}</td></tr>
            <tr><td>Recall</td><td>{{ recall }}</td></tr>
            <tr><td>F1-Score</td><td>{{ f1 }}</td></tr>
        </table>
        <h3>Confusion Matrix</h3>
        <table>
            <tr><th></th>{% for col in range(confusion_matrix.shape[1]) %}<th>Predicted {{ col }}</th>{% endfor %}</tr>
            {% for row in range(confusion_matrix.shape[0]) %}
            <tr><th>True {{ row }}</th>{% for col in range(confusion_matrix.shape[1]) %}<td>{{ confusion_matrix[row, col] }}</td>{% endfor %}</tr>
            {% endfor %}
        </table>
    </body>
    </html>
    ''', **metrics)

def update_metrics(y_true, y_pred):
    global metrics
    report = classification_report(y_true, y_pred, output_dict=True)
    metrics['accuracy'] = report['accuracy']
    metrics['precision'] = report['weighted avg']['precision']
    metrics['recall'] = report['weighted avg']['recall']
    metrics['f1-score'] = report['weighted avg']['f1-score']
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)

def load_and_resize_image(img_path, size=(224, 224)):
    try:
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
        img = cv2.resize(img, size)
        ret, buffer = cv2.imencode('.jpg', img)
        if not ret:
            raise ValueError(f"Failed to encode image: {img_path}")
        return buffer.tobytes()
    except Exception as e:
        print(f"Error processing image {img_path}: {e}")
        return None

def extract_label_from_filename(filename):
    if 'COVID19' in filename:
        return 0
    elif 'NORMAL' in filename:
        return 1
    else:
        raise ValueError(f"Unrecognized label in filename: {filename}")

def stream_folder(folder_path):
    image_files = sorted([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])
    y_true, y_pred = [], []
    for image_file in image_files:
        img_path = os.path.join(folder_path, image_file)
        frame = load_and_resize_image(img_path)
        if frame:
            img_array = np.array(Image.open(BytesIO(frame)).resize((224, 224)))
            img_array = np.expand_dims(img_array, axis=0)
            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions, axis=1)
            label = extract_label_from_filename(image_file)
            y_true.append(label)
            y_pred.append(predicted_class[0])

            # Update metrics after every batch or a certain number of images
            if len(y_true) >= 10:
                update_metrics(y_true, y_pred)
                y_true, y_pred = [], []

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(1)

def generate(folders):
    while True:
        with ThreadPoolExecutor(max_workers=len(folders)) as executor:
            future_to_path = {executor.submit(stream_folder, folder): folder for folder in folders}
            for future in future_to_path:
                try:
                    for frame in future.result():
                        yield frame
                except Exception as e:
                    print(f"Error in processing folder {future_to_path[future]}: {e}")

@app.route('/image_feed')
def image_feed():
    paths = os.getenv("IMAGE_PATHS", "").split(",")
    if not paths or paths == ['']:
        paths = [
            'C:/Users/Akoba/Desktop/START up/Covid19_research_project/data/raw/xray_images/LungData/train/COVID19',
            'C:/Users/Akoba/Desktop/START up/Covid19_research_project/data/raw/xray_images/LungData/train/NORMAL',
            'C:/Users/Akoba/Desktop/START up/Covid19_research_project/data/raw/xray_images/LungData/val/COVID19',
            'C:/Users/Akoba/Desktop/START up/Covid19_research_project/data/raw/xray_images/LungData/val/NORMAL',
            'C:/Users/Akoba/Desktop/START up/Covid19_research_project/data/raw/xray_images/LungData/test/COVID19',
            'C:/Users/Akoba/Desktop/START up/Covid19_research_project/data/raw/xray_images/LungData/test/NORMAL'
        ]
    return Response(stream_with_context(generate(paths)), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file found in the request'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    save_path = '/path/to/save/uploads/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    file_path = os.path.join(save_path, file.filename)
    img = Image.open(file)
    img = img.resize((224, 224))
    img.save(file_path)

    img_array = np.expand_dims(np.array(img), axis=0)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)

    return jsonify({'message': 'Image received and processed successfully', 'predicted_class': int(predicted_class[0])}), 200

def signal_handler(sig, frame):
    print("Shutting down gracefully...")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)
