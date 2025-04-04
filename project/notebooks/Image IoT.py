from flask import Flask, Response, stream_with_context
import time
import cv2
import os

app = Flask(__name__)

def generate(folders):
    while True:
        for folder_path in folders:
            # Get a list of image files in the folder
            image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
            image_files.sort()  # Ensure consistent order

            for image_file in image_files:
                img_path = os.path.join(folder_path, image_file)
                img = cv2.imread(img_path)
                # Encode the frame in JPEG format
                ret, buffer = cv2.imencode('.jpg', img)
                # Convert to bytes and yield it as a response
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                time.sleep(10)  # simulate delay to mimic real-time streaming

@app.route('/image_feed')
def image_feed():
    # Define the paths to your directories
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
    app.run(host='0.0.0.0', port=5050, debug=True)
