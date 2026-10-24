# Face Recognition App

A simple Streamlit app that recognizes faces. I trained it on the AT&T/ORL face dataset (40 people, 10 photos each), and you can also add your own photos.

## What it does
- Trains a face recognizer on the dataset (OpenCV: Haar detection + LBPH).
- Shows hold-out accuracy (about 97% on ORL in my test).
- Recognizes faces from an uploaded photo or the camera, and shows "Unknown" for faces it doesn't know.

## Run
pip install -r requirements.txt
streamlit run app.py

## Live app
<yahan apna Streamlit link paste karo>

## Limits
- Works best with front-facing faces and similar lighting.
- Uploaded photos and the trained model are lost when the app restarts.

Dataset credit: AT&T Laboratories Cambridge.
