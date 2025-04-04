from flask import Flask, Response, stream_with_context
from concurrent.futures import ThreadPoolExecutor
import time
import cv2
import os
import signal
import sys
import asyncio

app = Flask(__name__)

def signal_handler(sig, frame):
    print("Thanks, Machine is dead now...")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

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


def stream_folder(folder_path):
    frames = []
    image_files = sorted([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])
    for image_file in image_files:
        img_path = os.path.join(folder_path, image_file)
        frame = load_and_resize_image(img_path)
        if frame:
            frames.append((b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'))
        time.sleep(1)
    return frames

async def generate(folders):
    while True:
        with ThreadPoolExecutor(max_workers=len(folders)) as executor:
            future_to_path = {executor.submit(stream_folder, folder): folder for folder in folders}
            for future in future_to_path:
                for frame in future.result():
                    yield frame
                await asyncio.sleep(1)  # non-blocking sleep

@app.route('/image_feed')
def image_feed():
    # Define the paths to your directories
    paths = os.getenv("IMAGE_PATHS", "").split(",")  # Paths can be set in environment variables
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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)