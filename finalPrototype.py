import cv2
import numpy as np
from ultralytics import YOLO
import math
import cv2
import os

yoloCharacter = YOLO('YOLO/css.pt')
yoloPlate = YOLO('YOLO/license.pt')
yoloHelmet = YOLO('YOLO/hel.pt')


def firstCrop(img, predictions):
    first_crop = None  # Initialize first_crop outside the loop
    offset = (0, 0)    # Initialize offset
    initial_license_plate_box = (0, 0, 0, 0)  # Default value

    # Iterate through the bounding boxes
    for result in predictions:
        license_plate_boxes = result.boxes.data.cpu().numpy()
        for i, box in enumerate(license_plate_boxes):
            x1, y1, x2, y2, conf, cls = box
            # Round coordinates to the higher value and convert to integers
            x1, y1, x2, y2 = map(lambda val: int(math.ceil(val)), [x1, y1, x2, y2])

            # Save the initial coordinates of the license plate bounding box
            initial_license_plate_box = (x1, y1, x2, y2)

            # Draw a rectangle around the license plate on the original image        
            # Crop the image using array slicing
            first_crop = img[y1:y2, x1:x2].copy()

    # Calculate the offset
    offset = (initial_license_plate_box[0], initial_license_plate_box[1])

    return first_crop, offset

# Open the input video file
cap = cv2.VideoCapture('testvids/b5.MOV')

# Get the video frame width and height
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Create an output video file

# Assuming you already have cap, frame_width, and frame_height defined

# Specify the new output directory and filename
output_directory = 'output/'
output_filename = 'output.avi'

# Concatenate the directory and filename to create the full output path
output_path = os.path.join(output_directory, output_filename)

# Check if the directory exists, and create it if it doesn't
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Create an output video file
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Use the same codec as the input video
out = cv2.VideoWriter(output_path, fourcc, cap.get(5), (frame_width, frame_height))


# Initialize offset
offset = (0, 0)

counter = 0

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break
    helmet_predictions = yoloHelmet.predict(frame)
    license_predictions = yoloPlate.predict(frame)
    frame_with_license_box, offset = firstCrop(frame.copy(), license_predictions)
    character_predictions = yoloCharacter.predict(frame_with_license_box)
    # Draw bounding boxes and labels for license plate predictions
    for result in license_predictions:
        for pred in result.boxes.data.cpu().numpy():
            x1, y1, x2, y2, conf, cls = pred
            if conf > 0.5:  # Check confidence score
                x1, y1, x2, y2 = map(lambda val: int(math.ceil(val)), [x1, y1, x2, y2])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    for result in helmet_predictions:
        for pred in result.boxes.data.cpu().numpy():
            x1, y1, x2, y2, conf, cls = pred
            if conf > 0.5:  # Check confidence score
                x1, y1, x2, y2 = map(lambda val: int(math.ceil(val)), [x1, y1, x2, y2])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = yoloHelmet.names[int(cls)]
                cv2.putText(frame, f"{label}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Draw bounding boxes and labels for character predictions
    for result in character_predictions:
        for pred in result.boxes.data.cpu().numpy():
            x1, y1, x2, y2, conf, cls = pred
            if conf > 0.5:  # Check confidence score
                x1, y1, x2, y2 = map(lambda val: int(math.ceil(val)), [x1, y1, x2, y2])
            # Add the offset to convert coordinates to total video coordinates
                x1 += offset[0]
                y1 += offset[1]
                x2 += offset[0]
                y2 += offset[1]

                # Draw rectangle and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                label = yoloCharacter.names[int(cls)]
                cv2.putText(frame, f"{label}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Write the frame to the output video
    out.write(frame)

    # Display the frame
    cv2.imshow("Output", frame)

    counter += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and writer objects
cap.release()
out.release()

# Close all windows
cv2.destroyAllWindows()
