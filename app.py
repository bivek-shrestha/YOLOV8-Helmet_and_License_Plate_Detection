from flask import Flask, render_template, request, redirect, url_for
import cv2
import numpy as np
from ultralytics import YOLO
import math
import os
import tempfile
import imageio

app = Flask(__name__,static_folder='static')

yoloCharacter = YOLO('YOLO/css.pt')
yoloPlate = YOLO('YOLO/LPP.pt')
yoloHelmet = YOLO('YOLO/hel.pt')

def firstCrop(img, predictions):
    first_crop = None
    offset = (0, 0)
    initial_license_plate_box = (0, 0, 0, 0)

    for result in predictions:
        license_plate_boxes = result.boxes.data.cpu().numpy()
        for i, box in enumerate(license_plate_boxes):
            x1, y1, x2, y2, conf, cls = box
            x1, y1, x2, y2 = map(lambda val: int(math.ceil(val)), [x1, y1, x2, y2])

            initial_license_plate_box = (x1, y1, x2, y2)

            first_crop = img[y1:y2, x1:x2].copy()

    offset = (initial_license_plate_box[0], initial_license_plate_box[1])

    return first_crop, offset

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_video', methods=['POST'])
def process_video():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        file.save(temp_file.name)
        input_video_path = temp_file.name

    output_video_path = process_video_file(input_video_path)

    return render_template('result.html')

def process_video_file(input_video_path):
    cap = cv2.VideoCapture(input_video_path)

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    fourcc = cv2.VideoWriter_fourcc(*'h264')
    output_video_path = os.path.join('static', 'output.mp4')
    out = cv2.VideoWriter(output_video_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        helmet_predictions = yoloHelmet.predict(frame)
        license_predictions = yoloPlate.predict(frame)
        frame_with_license_box, offset = firstCrop(frame.copy(), license_predictions)
        character_predictions = yoloCharacter.predict(frame_with_license_box)

        for result in license_predictions:
            for pred in result.boxes.data.cpu().numpy():
                x1, y1, x2, y2, conf, cls = pred
                if conf > 0.5:
                    x1, y1, x2, y2 = map(lambda val: int(math.ceil(val)), [x1, y1, x2, y2])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        for result in helmet_predictions:
            for pred in result.boxes.data.cpu().numpy():
                x1, y1, x2, y2, conf, cls = pred
                if conf > 0.5:
                    x1, y1, x2, y2 = map(lambda val: int(math.ceil(val)), [x1, y1, x2, y2])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = yoloHelmet.names[int(cls)]
                    cv2.putText(frame, f"{label}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        for result in character_predictions:
            for pred in result.boxes.data.cpu().numpy():
                x1, y1, x2, y2, conf, cls = pred
                if conf > 0.5:
                    x1, y1, x2, y2 = map(lambda val: int(math.ceil(val)), [x1, y1, x2, y2])
                    x1 += offset[0]
                    y1 += offset[1]
                    x2 += offset[0]
                    y2 += offset[1]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    label = yoloCharacter.names[int(cls)]
                    cv2.putText(frame, f"{label}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        out.write(frame)

    cap.release()
    out.release()

    return output_video_path

if __name__ == '__main__':
    app.run(debug=True)
