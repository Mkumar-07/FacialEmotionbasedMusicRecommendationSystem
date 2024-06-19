import gradio as gr
import tensorflow as tf
import numpy as np
import pandas as pd
import requests
import keras

from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.preprocessing import image
from PIL import Image
import cv2

model = keras.saving.load_model('./Final_ResNet50_Model.keras')
songs = pd.read_csv('./songs_mood.csv')

emotion_labels = {'angry': 0, 'fear': 1, 'happy': 2, 'neutral': 3, 'sad': 4}
index_to_emotion = {v: k for k, v in emotion_labels.items()}
emotion_reference = {'happy':'Happy','sad':'Sad','neutral':'Calm','angry':'Energetic','fear':'Energetic'}

def prepare_image(img_pil):
  img = img_pil.resize((224,224))
  img_array = img_to_array(img)
  img_array = np.expand_dims(img_array, axis=0)  # Convert single image to a batch.
  img_array /= 255.0
  return img_array

def predict_emotion(image):
  processed_image = prepare_image(image)
  prediction = model.predict(processed_image)
  predicted_class = np.argmax(prediction, axis=1)
  predicted_emotion = index_to_emotion.get(predicted_class[0],'Unknown Class')
  return predicted_emotion

def recommend_music(image):
  # predict emotion
  emotion = predict_emotion(image)
  emotion_ref = emotion_reference[emotion]

  # fetch songs list corresponding to emotions
  songs_list = songs[songs['mood'] == emotion_ref]
  songs_list = songs_list.sample(n=5)
  songs_list = songs_list.reset_index(drop=True)

  # converting songs list to hyperlinks to send it to output
  songs_html = "<ul>"
  for x, row in songs_list.iterrows():
    song_url = f'spotify:track:{row["id"]}'
    song_name = f"{row['name']} by {row['artist']}"
    songs_html += f"<li><a href='{song_url}'>{song_name}</a></li>"
  songs_html += "</ul>"
  return emotion, songs_html

upload_interface = gr.Interface(
    fn = recommend_music,
    inputs = gr.Image(type='pil',label = 'Image',sources=["upload","clipboard"]),
    outputs=[gr.Textbox(label='Emotion'), gr.HTML(label='Recommended Songs')],
    title = 'VibeSync - Facial Emotion Based Music Recommendation System',
    description = "Here you can upload your image and we will identify your mood and recommend songs based on that"
)

upload_interface.launch(debug = True)