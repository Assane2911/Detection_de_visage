import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

st.set_page_config(page_title="Détection de Visages", layout="wide")

st.title("Détection de visages (image upload) ")
st.write("Téléverse une image — l'app détecte et encadre les visages.")

# Charger cascade
haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(haar_path)

uploaded_file = st.file_uploader("Choisir une image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # lire l'image en PIL puis en OpenCV
    image = Image.open(uploaded_file).convert("RGB")
    img = np.array(image)[:, :, ::-1].copy()  # RGB -> BGR

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # dessiner rectangles
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (10, 200, 255), 2)
        cv2.putText(img, "Visage", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    # convertir pour affichage streamlit (BGR -> RGB)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    st.image(img_rgb, caption=f"Visages détectés : {len(faces)}", use_container_width=True)

    # proposer téléchargement de l'image annotée
    buffer = io.BytesIO()
    Image.fromarray(img_rgb).save(buffer, format="PNG")
    st.download_button("Télécharger l'image annotée", data=buffer, file_name="annotated.png", mime="image/png")
else:
    st.info("Téléverse une image pour commencer.")
