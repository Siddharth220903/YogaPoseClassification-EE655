# EE655 Project: Real-Time Yoga Pose Guidance System\n
This project uses computer vision and AI to analyze your yoga practice in real-time via webcam. It provides guidance for five fundamental poses (Downward Dog, Warrior, Tree, Goddess, Plank) by:
Extracting skeletal landmarks using MediaPipe.
Classifying the pose with a Custom CNN (achieving 99.62% accuracy).
Generating personalized alignment feedback using joint angles and a Large Language Model (LLM).
The goal is to offer accessible, immediate feedback for safer and more effective self-practice.


## Data: \n
We utilize the publicly available Yoga Poses dataset from Kaggle(https://www.kaggle.com/datasets/niharika41298/yoga-poses-dataset), which contains labeled images of five foundational yoga poses: Downward Dog, Goddess, Tree, Plank,
and Warrior. The dataset contains 1551 images and is organized into training and testing set with a split of 70% training data and 30% testing data, with a propotional split of the
5 yoga classes. As the images were scraped from the internet using Bing’s API, they exhibit varying levels of quality, including potential noise such as watermarks, text overlays,
and inconsistent backgrounds.
## Model Training:
## Inference 
