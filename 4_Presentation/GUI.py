import streamlit as st
import pandas as pd
from PIL import Image
import tensorflow as tf
import numpy as np
import time
import io

st.title("Wheat Disease Detection")

col1, col2, col3 =st.columns ([1,2,1])

model = tf.keras.models.load_model('baseline_model_01.keras')
class_names = ['class1', 'class2', 'class3','class4','class5','class6','class7','class8','class9','class10']  # Beispiel-Klassenbezeichnungen

# Session State initialisieren, falls noch nicht vorhanden
if 'history' not in st.session_state:
    st.session_state['history'] = []


uploaded_file = col2.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
uploaded_photo = col2.camera_input("or take a photo")
progress_bar = col2.progress(0)

for perc_completed in range (100):
    time.sleep(0.05)
    progress_bar.progress(perc_completed+1)
col2.success("Photo uploaded successfully")



if uploaded_file is not None:
    # Bild anzeigen
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Bild verarbeiten und vorhersagen
    img = img.resize((256, 256))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    result= (f"Prediction: {class_names[np.argmax(score)]} ({100 * np.max(score):.2f}%)")


# Ergebnis anzeigen
    col2.write(result)

# Historie aktualisieren
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    st.session_state['history'].append((img_bytes.getvalue(), result))

# Sidebar mit Historie der Bilder und Ergebnisse
st.sidebar.title("History")
for i, (img_data, result) in enumerate(st.session_state['history']):
    st.sidebar.write(f"Image {i+1}:")
    st.sidebar.image(Image.open(io.BytesIO(img_data)), width=100)
    st.sidebar.write(result)

