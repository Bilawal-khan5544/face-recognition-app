import os

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from faceid import core

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "sample_data")

st.set_page_config(page_title="Face Recognition", page_icon="🧑", layout="wide")


@st.cache_resource(show_spinner="Loading sample dataset…")
def sample_gallery():
    return core.load_folder_dataset(SAMPLE_DIR) if os.path.isdir(SAMPLE_DIR) else {}


ss = st.session_state
ss.setdefault("user_gallery", {})
ss.setdefault("rec", None)
ss.setdefault("report", None)
ss.setdefault("last_shot", None)


def active_gallery():
    return core.merge(sample_gallery() if ss.get("use_sample", True) else {}, ss.user_gallery)


st.title("🧑 Face Recognition")
st.caption("Haar face detection + LBPH recognition (OpenCV). Add a dataset → train → recognize. Images are processed in memory only.")

with st.sidebar:
    st.header("Dataset")
    st.checkbox("Include sample dataset (AT&T/ORL, 40 people × 10 photos)", value=True, key="use_sample")
    g = active_gallery()
    st.metric("People", len(g))
    st.metric("Photos", sum(len(v) for v in g.values()))
    if st.button("Clear my uploads / enrollments"):
        ss.user_gallery, ss.rec, ss.report = {}, None, None
        st.rerun()
    st.divider()
    st.caption("Sample data: AT&T Laboratories Cambridge (ORL) face database.")

tab_data, tab_train, tab_rec = st.tabs(["1 · Dataset", "2 · Train & evaluate", "3 · Recognize"])

# ---- 1. dataset ---------------------------------------------------------------
with tab_data:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Upload a dataset (.zip)")
        st.write("Folder-per-person: `person_a/1.jpg, person_a/2.jpg, person_b/1.jpg …` (≥ 5 photos each works best).")
        crop = st.checkbox("Photos are already cropped faces (skip detection)", value=False)
        zf = st.file_uploader("Zip file", type="zip", key="zipfile")
        if zf and st.button("Add zip to dataset"):
            try:
                added, skipped = core.zip_to_gallery(zf.getvalue(), detect=not crop)
                ss.user_gallery = core.merge(ss.user_gallery, added)
                st.success(f"Added {sum(len(v) for v in added.values())} photos of {len(added)} people.")
                if skipped:
                    st.warning(f"Skipped {len(skipped)} file(s) (no face found / too large / not in a folder).")
            except Exception as exc:  # corrupt zip etc.
                st.error(f"Could not read the zip: {exc}")
    with c2:
        st.subheader("Enroll a person")
        name = st.text_input("Name", key="enroll_name").strip()
        files = st.file_uploader("Photos", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="enroll_files")
        if st.button("Add photos") and name and files:
            n_ok = 0
            for f in files:
                img = core.decode_image(f.getvalue())
                faces = core.extract_faces(img) if img is not None else []
                if len(faces) >= 1:
                    ss.user_gallery.setdefault(name, []).append(faces[0][1])
                    n_ok += 1
            st.success(f"Added {n_ok}/{len(files)} photos for {name}.") if n_ok else st.error("No face found in the photos.")
        shot = st.camera_input("…or take a photo", key="enroll_cam")
        if shot and name and shot.getvalue() != ss.last_shot:
            ss.last_shot = shot.getvalue()
            faces = core.extract_faces(core.decode_image(shot.getvalue()))
            if len(faces) == 1:
                ss.user_gallery.setdefault(name, []).append(faces[0][1])
                st.success(f"Saved photo #{len(ss.user_gallery[name])} for {name}. Take a few more (angles, lighting).")
            else:
                st.error(f"Need exactly one face in the photo (found {len(faces)}).")
        elif shot and not name:
            st.info("Type a name first.")
    if ss.user_gallery:
        st.subheader("Your additions")
        st.dataframe(pd.DataFrame({"person": list(ss.user_gallery), "photos": [len(v) for v in ss.user_gallery.values()]}))

# ---- 2. train -----------------------------------------------------------------
with tab_train:
    g = active_gallery()
    thin = [k for k, v in g.items() if len(v) < 3]
    if thin:
        st.warning(f"Few photos (<3) for: {', '.join(thin[:8])}{'…' if len(thin) > 8 else ''}. Accuracy will suffer.")
    if st.button("Train model", type="primary", disabled=len(g) < 2):
        with st.spinner("Training…"):
            report = core.evaluate(g)
            rec = core.FaceRecognizer().train(g)
            rec.threshold = core.auto_threshold(report["genuine_distances"])
            ss.rec, ss.report = rec, report
    if len(g) < 2:
        st.info("Add at least 2 people (tick the sample dataset or upload your own).")
    if ss.rec is not None and ss.report:
        r = ss.report
        m1, m2, m3 = st.columns(3)
        m1.metric("People", len(ss.rec.names))
        m2.metric("Hold-out accuracy", f"{r['accuracy']:.1%}" if r["accuracy"] is not None else "n/a",
                  help="Last 2 photos of each person were held out, model trained on the rest.")
        m3.metric("Auto 'Unknown' threshold", f"{ss.rec.threshold:.0f}")
        if r["per_person"]:
            weak = sorted(r["per_person"].items(), key=lambda kv: kv[1])[:10]
            st.write("Weakest people (hold-out accuracy):")
            st.dataframe(pd.DataFrame(weak, columns=["person", "accuracy"]))

# ---- 3. recognize -------------------------------------------------------------
with tab_rec:
    if ss.rec is None:
        st.info("Train the model first (tab 2).")
    else:
        thr = st.slider("'Unknown' threshold (lower = stricter)", 20, 200, int(ss.rec.threshold))
        whole = st.checkbox("If no face is detected, treat the whole image as a face (cropped photos)", value=False)
        src = st.radio("Image source", ["Upload", "Camera"], horizontal=True)
        data = None
        if src == "Upload":
            up = st.file_uploader("Image", type=["jpg", "jpeg", "png", "pgm", "bmp"], key="rec_up")
            data = up.getvalue() if up else None
        else:
            cam = st.camera_input("Take a photo", key="rec_cam")
            data = cam.getvalue() if cam else None
        if data:
            img = core.decode_image(data)
            if img is None:
                st.error("Could not read that image.")
            else:
                res = core.recognize_image(ss.rec, img, thr, whole)
                if not res:
                    st.warning("No face detected. Try a clearer, front-facing photo.")
                else:
                    st.image(cv2.cvtColor(core.annotate(img, res), cv2.COLOR_BGR2RGB))
                    st.dataframe(pd.DataFrame([{"person": r["name"], "distance": round(r["distance"], 1)} for r in res]))
                    st.caption("Distance: lower = more similar. Above the threshold → Unknown.")
