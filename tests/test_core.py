import io
import os
import zipfile

import cv2
import numpy as np
import pytest

from faceid import core

SAMPLE = os.path.join(os.path.dirname(__file__), "..", "sample_data")


@pytest.fixture(scope="module")
def gallery():
    return core.load_folder_dataset(SAMPLE)


def test_sample_dataset_loaded(gallery):
    assert len(gallery) == 40 and all(len(v) == 10 for v in gallery.values())
    assert gallery["person_01"][0].shape == (core.SIZE, core.SIZE)


def test_holdout_accuracy_is_high(gallery):
    rep = core.evaluate(gallery)
    assert rep["accuracy"] > 0.9 and rep["n_test"] == 80
    assert 40 <= core.auto_threshold(rep["genuine_distances"]) <= 200


def test_train_predict_and_unknown(gallery):
    known = {k: v[:8] for k, v in list(gallery.items())[:30]}
    rec = core.FaceRecognizer().train(known)
    name, dist = rec.predict(gallery["person_01"][9], threshold=1e9)
    assert name == "person_01" and dist > 0
    assert rec.predict(gallery["person_01"][9], threshold=0.001)[0] == "Unknown"
    with pytest.raises(ValueError):
        core.FaceRecognizer().train({"only_one": known["person_01"]})


def test_zip_to_gallery_and_recognize_image(gallery):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for p in ("person_01", "person_02", "person_03"):
            for i in range(1, 6):
                img = cv2.imread(f"{SAMPLE}/{p}/{i:02d}.png", cv2.IMREAD_GRAYSCALE)
                z.writestr(f"dataset/{p}/{i}.png", cv2.imencode(".png", img)[1].tobytes())
        z.writestr("dataset/readme.txt", "ignored")
        z.writestr("loose.png", b"not an image")
    g, skipped = core.zip_to_gallery(buf.getvalue(), detect=False)
    assert sorted(g) == ["person_01", "person_02", "person_03"] and all(len(v) == 5 for v in g.values())
    assert skipped == ["loose.png"]
    rec = core.FaceRecognizer().train(g)
    img = cv2.imread(f"{SAMPLE}/person_02/09.png")
    res = core.recognize_image(rec, img, 1e9, whole_if_none=True)
    assert res and res[0]["name"] == "person_02"
    assert core.annotate(img, res).shape == img.shape


def test_decode_and_detect_handle_bad_input():
    assert core.decode_image(b"") is None and core.decode_image(b"garbage") is None
    assert core.detect_faces(np.zeros((200, 200), np.uint8)) == []
