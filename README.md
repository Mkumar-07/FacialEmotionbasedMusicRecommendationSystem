# VibeSync - Facial Emotion Based Music Recommendation System

## Overview

VibeSync is a facial emotion-based music recommendation system that identifies the mood of a user from their uploaded image and recommends songs accordingly. By leveraging the power of deep learning and a pre-trained ResNet50 model, VibeSync can detect emotions and provide personalized music suggestions.

## Features

- Upload an image to detect your emotion.
- Receive song recommendations based on your detected emotion.
- Simple and intuitive web interface powered by Gradio.

## Requirements

- Python 3.6 or higher
- Gradio
- TensorFlow
- Keras
- NumPy
- Pandas
- Requests
- PIL (Pillow)
- OpenCV

## Setup

1. Clone the repository:
    ```bash
    git clone https://github.com/Mkumar-07/FacialEmotionbasedMusicRecommendationSystem.git
    cd FacialEmotionbasedMusicRecommendationSystem
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Ensure you have the trained model file (`Final_ResNet50_Model.keras`) and the songs dataset (`songs_mood.csv`) in the project directory.

## Usage

Run the application using the following command:
```bash
python app.py
```

This will launch a Gradio web interface where you can upload an image and get song recommendations based on the detected emotion.

## Code Explanation

### Imports and Setup

The script begins by importing necessary libraries such as Gradio, TensorFlow, NumPy, Pandas, Requests, Keras, PIL, and OpenCV.

### Model and Data Loading

- The pre-trained ResNet50 model is loaded from the file `Final_ResNet50_Model.keras`.
- The song dataset `songs_mood.csv` is loaded into a Pandas DataFrame.

### Helper Functions

- `prepare_image(img_pil)`: Resizes the input image to 224x224, converts it to an array, normalizes the pixel values, and adds an extra dimension to make it compatible with the model input.
- `predict_emotion(image)`: Uses the loaded model to predict the emotion from the prepared image and returns the predicted emotion label.
- `recommend_music(image)`: Predicts the emotion from the image, selects a list of songs corresponding to the detected emotion, and returns the emotion and a list of song recommendations as HTML links.

### Gradio Interface

The Gradio interface is defined to accept an image input, process it through the `recommend_music` function, and display the detected emotion and song recommendations.

## Conclusion

VibeSync provides an innovative way to recommend music based on the user's current mood detected from their facial expression. It's a fun and interactive application that combines computer vision and music recommendation systems.
