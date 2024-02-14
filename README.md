# YOLOV8-Helmet_and_License_Plate_Detection


Python, YOLOv8

# Installation
## Description

The project focuses on Nepali Number Plate Recognition and Helmet Detection. Leveraging advanced machine learning, we have made significant strides in achieving precise number plate recognition. The project also includes pioneering efforts in dataset preprocessing for robust helmet detection models, contributing to enhanced enforcement of helmet usage regulations.

Key accomplishments include the successful implementation of an OCR system for Nepali number plates and comprehensive dataset preprocessing for helmet detection. Looking ahead, our project envisions expanding model capabilities, exploring real-time video integration, and consistently refining system accuracy.



## Prerequisites

Ensure you have the following installed before setting up the project:

- Python 3.x
- pip (Python package installer)
- Git

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/your-project](https://github.com/bivek-shrestha/YOLOV8-Helmet_and_License_Plate_Detection.git
cd YOLOV8-Helmet_and_License_Plate_Detection
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

The `requirements.txt` file should contain the following dependencies:

```plaintext
numpy
opencv-python
torch>=1.7.0
tqdm
yolov8
ultralytics
```

### 3. Set Up YOLOv8

Follow the YOLOv8 installation guide if additional setup is required:

[Yolov8 Installation Guide](https://github.com/ultralytics/yolov5)

### 4. Set Up Roboflow (Optional)

If you are using Roboflow for data collection and management, sign up for an account and follow their setup instructions.

[Roboflow Website](https://roboflow.com/)

## Usage

### 1. Data Collection and Management

Follow the Ultralytics documentation for managing datasets:

[Ultralytics Datasets](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data)

### 2. Training YOLOv8 Model

Run the following command to train the YOLOv8 model:

```bash
python train.py --data path/to/dataset.yaml --cfg models/yolov8.yaml --weights '' --batch-size 16
```

Modify the command based on your dataset path, configuration file, and batch size.

### 3. Inference

Run the following command to perform inference on an image:

```bash
python detect.py --source path/to/image.jpg --weights path/to/weights.pt
```

Modify the command based on your input image and trained weights and their directory locations.

## Acknowledgement

I would like to thank our advisors for the proper guidance and support throughout the project. We would also convey deep gratitude to DR. Dibakar Raj Pant Phd and Er. Jit pant from Infodevelopers for their help and support in this project. It would have been a very difficult path without their effort.
I am indebted to our advisors without whom this project would not have been successful.





# Methodology

YOLOv8s model trained on our custom dataset detects motorcycle, car, bus, truck, license plate, and helmets. The detected objects will be  tracked using DeepSORT.The license plate is fed to the character segmentation module, and the segmented characters are sent for recognition. The output given by the character recognition model is annotated along with the plate in the video frame. 



<img width="230" alt="Screenshot 2024-02-14 175853" src="https://github.com/bivek-shrestha/YOLOV8-Helmet_and_License_Plate_Detection/assets/155466197/5f0266d8-6fa7-4257-81f2-2cd3a61f258b">



# Outputs



 <img width="642" alt="image" src="https://github.com/bivek-shrestha/YOLOV8-Helmet_and_License_Plate_Detection/assets/155466197/e8c8987d-ec30-420e-9992-227f62929f52">




https://github.com/bivek-shrestha/YOLOV8-Helmet_and_License_Plate_Detection/assets/155466197/ba46f309-2541-4e8a-b334-6ceaf0be857b


 

