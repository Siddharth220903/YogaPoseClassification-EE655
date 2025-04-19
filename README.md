# EE655 Project: Real-Time Yoga Pose Guidance System
This project uses computer vision and AI to analyze your yoga practice in real-time via webcam. It provides guidance for five fundamental poses (Downward Dog, Warrior, Tree, Goddess, Plank) by:
Extracting skeletal landmarks using MediaPipe.
Classifying the pose with a Custom CNN (achieving 99.62% accuracy).
Generating personalized alignment feedback using joint angles and a Large Language Model (LLM).
The goal is to offer accessible, immediate feedback for safer and more effective self-practice.


## Dataset Loading:
We utilize the publicly available Yoga Poses dataset from Kaggle(https://www.kaggle.com/datasets/niharika41298/yoga-poses-dataset), which contains labeled images of five foundational yoga poses: Downward Dog, Goddess, Tree, Plank,
and Warrior. The dataset contains 1551 images and is organized into training and testing set with a split of 70% training data and 30% testing data, with a propotional split of the
5 yoga classes. As the images were scraped from the internet using Bing’s API, they exhibit varying levels of quality, including potential noise such as watermarks, text overlays,
and inconsistent backgrounds.

Before training the model, download the data in a folder. Maintain seperate directories for Train and Test.

## Model Training:
To train the model, use [Train.py](Train.py). Update the dataset paths. To run the file using the following command: python train.py. Save the model weights to run the inference file.
You can change the model defination in the code, to train on different models. 

## Inference:
[Inference.py](Inference.py) Inference script helps enable live Yoga pose classification using your webcam. Update API key to google's gemini Flash 1.5 to run the inference script and generate suggestions to improve your Yoga Aasan. Update the path to the weights of the trained model.
