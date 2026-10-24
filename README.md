# Face Recognition (Streamlit)

Dataset → train → recognize, in the browser. OpenCV only (Haar face detection + LBPH recognizer): no model downloads, no GPU, light enough for Streamlit Community Cloud.

- **Dataset:** AT&T/ORL face database (40 people × 10 photos) is bundled in `sample_data/` (credit: AT&T Laboratories Cambridge). You can also upload your own zip (`person_name/photo.jpg` layout) or enroll people with the camera.
- **Evaluate:** last 2 photos of each person are held out; the app shows hold-out accuracy (~97% on ORL in testing) and an auto "Unknown" threshold.
- **Recognize:** upload or camera photo → boxes + names + distance (lower = more similar). Above the threshold → `Unknown`.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
pytest -q
```

## Deploy via GitHub → Streamlit Community Cloud
1. Create a GitHub repo and push this folder (keep `app.py`, `requirements.txt`, `faceid/`, `sample_data/` at the repo root).
   ```bash
   git init && git add . && git commit -m "Face recognition app"
   git branch -M main && git remote add origin https://github.com/<you>/<repo>.git && git push -u origin main
   ```
2. Go to https://share.streamlit.io → **Create app** → pick the repo, branch `main`, main file `app.py` → **Deploy**.
3. GitHub Actions (`.github/workflows/ci.yml`) runs the tests on every push.

## Limits (honest)
- LBPH is a classic method: it works best with frontal faces and similar lighting. For harder conditions use a deep model (e.g. ArcFace/SFace) — this app does not.
- Faces are matched against only the people you trained; strangers can still be mis-matched if they look similar. Tune the threshold slider.
- Streamlit Cloud storage is temporary: uploads and the trained model live in the session only and are lost on restart.
- Camera input needs HTTPS (Streamlit Cloud provides it) and browser permission.
- Use only photos you have the right to use; get consent before enrolling real people.
- Tested: unit tests + Streamlit `AppTest` smoke test with OpenCV 4.10 locally. Not tested on Streamlit Cloud itself or with a real webcam.
