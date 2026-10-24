"""Face detection (Haar cascade) + recognition (LBPH). Pure OpenCV: no model downloads, light enough for Streamlit Cloud."""
import io
import os
import zipfile
from collections import defaultdict

import cv2
import numpy as np

SIZE = 100
IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".pgm", ".webp")
MAX_ZIP_IMAGES = 2000
MAX_IMAGE_BYTES = 5_000_000
_cascade = None


def _get_cascade():
    global _cascade
    if _cascade is None:
        _cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    return _cascade


def decode_image(data: bytes):
    """Bytes -> BGR image, or None if unreadable."""
    arr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR) if arr.size else None


def to_gray(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def prep_face(gray: np.ndarray) -> np.ndarray:
    return cv2.equalizeHist(cv2.resize(gray, (SIZE, SIZE), interpolation=cv2.INTER_AREA))


def detect_faces(gray: np.ndarray, min_size: int = 60) -> list[tuple[int, int, int, int]]:
    boxes = _get_cascade().detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(min_size, min_size))
    return sorted((tuple(int(v) for v in b) for b in boxes), key=lambda b: -b[2] * b[3])


def extract_faces(img: np.ndarray, whole_if_none: bool = False) -> list[tuple[tuple, np.ndarray]]:
    """[(box, prepped_face)]. With whole_if_none, an image without a detected face is used as-is (pre-cropped datasets)."""
    gray = to_gray(img)
    boxes = detect_faces(gray)
    if not boxes and whole_if_none:
        boxes = [(0, 0, gray.shape[1], gray.shape[0])]
    return [((x, y, w, h), prep_face(gray[y:y + h, x:x + w])) for x, y, w, h in boxes]


# ---- datasets ---------------------------------------------------------------
def load_folder_dataset(root: str) -> dict[str, list[np.ndarray]]:
    """Folder-per-person layout: root/<person>/<images>. Images are treated as already-cropped faces."""
    gallery: dict[str, list] = {}
    for person in sorted(os.listdir(root)):
        folder = os.path.join(root, person)
        if not os.path.isdir(folder):
            continue
        faces = []
        for f in sorted(os.listdir(folder)):
            if f.lower().endswith(IMG_EXT):
                g = cv2.imread(os.path.join(folder, f), cv2.IMREAD_GRAYSCALE)
                if g is not None:
                    faces.append(prep_face(g))
        if faces:
            gallery[person] = faces
    return gallery


def zip_to_gallery(data: bytes, detect: bool = True) -> tuple[dict[str, list], list[str]]:
    """Reads a zip in memory (nothing is extracted to disk). Person name = the image's parent folder."""
    gallery: dict[str, list] = defaultdict(list)
    skipped: list[str] = []
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        files = [i for i in z.infolist() if not i.is_dir() and i.filename.lower().endswith(IMG_EXT)
                 and "__MACOSX" not in i.filename]
        for info in files[:MAX_ZIP_IMAGES]:
            parts = info.filename.replace("\\", "/").split("/")
            if len(parts) < 2 or info.file_size > MAX_IMAGE_BYTES:
                skipped.append(info.filename)
                continue
            img = decode_image(z.read(info))
            faces = extract_faces(img, whole_if_none=not detect) if img is not None else []
            if len(faces) < 1:
                skipped.append(info.filename)
                continue
            gallery[parts[-2]].append(faces[0][1])  # largest face
        if len(files) > MAX_ZIP_IMAGES:
            skipped.append(f"(+{len(files) - MAX_ZIP_IMAGES} images over the {MAX_ZIP_IMAGES} limit)")
    return dict(gallery), skipped


def merge(*galleries: dict) -> dict[str, list]:
    out: dict[str, list] = defaultdict(list)
    for g in galleries:
        for k, v in g.items():
            out[k].extend(v)
    return dict(out)


# ---- recognizer ---------------------------------------------------------------
class FaceRecognizer:
    def __init__(self):
        self.names: list[str] = []
        self.model = None
        self.threshold: float = 90.0

    def train(self, gallery: dict[str, list]) -> "FaceRecognizer":
        if len(gallery) < 2:
            raise ValueError("Need at least 2 people to train a recognizer")
        self.names = sorted(gallery)
        X, y = [], []
        for i, n in enumerate(self.names):
            X += gallery[n]
            y += [i] * len(gallery[n])
        self.model = cv2.face.LBPHFaceRecognizer_create(radius=2, neighbors=8, grid_x=8, grid_y=8, threshold=1e9)
        self.model.train(X, np.array(y))
        return self

    def predict(self, face: np.ndarray, threshold: float | None = None) -> tuple[str, float]:
        """(name or 'Unknown', distance). Lower distance = more similar."""
        label, dist = self.model.predict(face)
        limit = self.threshold if threshold is None else threshold
        return (self.names[label] if dist <= limit else "Unknown"), float(dist)


def evaluate(gallery: dict[str, list], test_per_person: int = 2) -> dict:
    """Hold out the last images of each person, train on the rest, report accuracy + genuine distances."""
    train, test = {}, []
    for n, faces in gallery.items():
        if len(faces) >= test_per_person + 2:
            train[n] = faces[:-test_per_person]
            test += [(n, f) for f in faces[-test_per_person:]]
        else:
            train[n] = faces
    if len(train) < 2 or not test:
        return {"accuracy": None, "n_test": 0, "per_person": {}, "genuine_distances": []}
    rec = FaceRecognizer().train(train)
    hits, per, dists = 0, defaultdict(lambda: [0, 0]), []
    for n, f in test:
        label, d = rec.model.predict(f)
        ok = rec.names[label] == n
        hits += ok
        per[n][0] += ok
        per[n][1] += 1
        if ok:
            dists.append(float(d))
    return {"accuracy": hits / len(test), "n_test": len(test),
            "per_person": {k: v[0] / v[1] for k, v in per.items()}, "genuine_distances": dists}


def auto_threshold(genuine: list[float]) -> float:
    """95th percentile of correct-match distances + 10% margin (clamped)."""
    if len(genuine) < 5:
        return 90.0
    return float(np.clip(np.percentile(genuine, 95) * 1.1, 40, 200))


def annotate(img: np.ndarray, results: list[dict]) -> np.ndarray:
    out = img.copy()
    for r in results:
        x, y, w, h = r["box"]
        color = (60, 200, 60) if r["name"] != "Unknown" else (60, 60, 230)
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        cv2.putText(out, f"{r['name']} ({r['distance']:.0f})", (x, max(15, y - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, color, 2, cv2.LINE_AA)
    return out


def recognize_image(rec: FaceRecognizer, img: np.ndarray, threshold: float, whole_if_none: bool = False) -> list[dict]:
    res = []
    for box, face in extract_faces(img, whole_if_none):
        name, dist = rec.predict(face, threshold)
        res.append({"box": box, "name": name, "distance": dist})
    return res
