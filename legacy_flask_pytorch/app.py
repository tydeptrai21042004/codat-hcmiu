import os
import io
from typing import Dict, Any, List, Optional

import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn.functional as F
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

from model_def import SmallResNet

# --------------------
# Config
# --------------------
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(APP_ROOT, "models")

BINARY_PTH = os.path.join(MODEL_DIR, "cnn_smallresnet_binary.pth")
MULTI_PTH  = os.path.join(MODEL_DIR, "cnn_smallresnet_multi.pth")

IMG_SIZE = 64  # must match training
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".webp"}

# Multi-class order must match your training class_names
CLASS_NAMES_7 = ["nv", "mel", "bkl", "bcc", "akiec", "vasc", "df"]

# Binary mapping in your doc:
# B=1: mel, bcc, akiec, vasc ; M=0: nv, df, bkl
BINARY_LABELS = ["M(0)=benign", "B(1)=malignant"]

# --------------------
# Device
# --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("[INFO] device:", device)

# --------------------
# Load models (if exist)
# --------------------
cnn_bin: Optional[SmallResNet] = None
cnn_multi: Optional[SmallResNet] = None

def load_model_if_exists(path: str, num_classes: int) -> Optional[SmallResNet]:
    if not os.path.exists(path):
        print(f"[WARN] Model not found: {path}")
        return None
    model = SmallResNet(num_classes=num_classes)
    sd = torch.load(path, map_location=device)
    model.load_state_dict(sd)
    model.to(device)
    model.eval()
    print(f"[INFO] Loaded model: {path}")
    return model

cnn_bin = load_model_if_exists(BINARY_PTH, num_classes=2)
cnn_multi = load_model_if_exists(MULTI_PTH, num_classes=7)

if cnn_bin is None and cnn_multi is None:
    raise FileNotFoundError(
        "Không tìm thấy model .pth trong thư mục models/. "
        "Hãy copy cnn_smallresnet_binary.pth và/hoặc cnn_smallresnet_multi.pth vào models/."
    )

# --------------------
# Flask app
# --------------------
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10MB limit


def preprocess_image(file_bytes: bytes) -> torch.Tensor:
    """
    Read image bytes -> RGB -> resize -> float32 [0,1] -> CHW tensor -> (1,3,H,W)
    """
    # PIL read
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = np.array(img)  # HWC RGB uint8

    # resize with cv2
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

    x = img.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))  # CHW
    x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)  # (1,3,H,W)
    return x.to(device)


@torch.no_grad()
def predict_binary(x: torch.Tensor) -> Dict[str, Any]:
    if cnn_bin is None:
        return {"available": False}

    logits = cnn_bin(x)  # (1,2)
    prob = F.softmax(logits, dim=1).cpu().numpy()[0]  # [p0, p1]
    pred = int(np.argmax(prob))

    return {
        "available": True,
        "pred_index": pred,
        "pred_label": BINARY_LABELS[pred],
        "prob_benign": float(prob[0]),
        "prob_malignant": float(prob[1]),
    }


@torch.no_grad()
def predict_multi(x: torch.Tensor, topk: int = 3) -> Dict[str, Any]:
    if cnn_multi is None:
        return {"available": False}

    logits = cnn_multi(x)  # (1,7)
    prob = F.softmax(logits, dim=1).cpu().numpy()[0]
    pred = int(np.argmax(prob))

    idxs = np.argsort(-prob)[:topk].tolist()
    top = [{"class": CLASS_NAMES_7[i], "prob": float(prob[i])} for i in idxs]

    return {
        "available": True,
        "pred_index": pred,
        "pred_label": CLASS_NAMES_7[pred],
        "topk": top,
        "probs": {CLASS_NAMES_7[i]: float(prob[i]) for i in range(len(CLASS_NAMES_7))}
    }


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html",
                           has_binary=cnn_bin is not None,
                           has_multi=cnn_multi is not None,
                           img_size=IMG_SIZE)


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"ok": False, "error": "No file field"}), 400

    f = request.files["file"]
    if f.filename == "":
        return jsonify({"ok": False, "error": "Empty filename"}), 400

    filename = secure_filename(f.filename)
    ext = os.path.splitext(filename.lower())[1]
    if ext not in ALLOWED_EXT:
        return jsonify({"ok": False, "error": f"Unsupported file type: {ext}"}), 400

    file_bytes = f.read()
    if not file_bytes:
        return jsonify({"ok": False, "error": "Empty file"}), 400

    try:
        x = preprocess_image(file_bytes)
        out_bin = predict_binary(x)
        out_mul = predict_multi(x, topk=3)

        return jsonify({
            "ok": True,
            "binary": out_bin,
            "multi": out_mul,
            "device": str(device),
            "img_size": IMG_SIZE
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)


