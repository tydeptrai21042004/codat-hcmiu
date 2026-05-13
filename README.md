# HAM10000 Light ML Pipeline — Vercel-ready

This version replaces the heavy Flask + PyTorch web stack with a lightweight full-stack web app:

- **Frontend:** Next.js + React + TypeScript
- **Backend:** Next.js Route Handlers under `app/api/*`
- **ML in web:** TensorFlow.js runs in the browser, so Vercel does not need to host PyTorch, scikit-learn, or CUDA dependencies
- **Dataset mode:** upload a small HAM10000 subset ZIP/CSV/images, or use the synthetic demo subset for UI testing
- **Download button:** exports model + reports as ZIP

> Education/research demo only. This is not a medical diagnostic tool.

---

## Why this stack is lighter for Vercel

The old Flask/PyTorch version is good for local GPU/Colab training, but it is heavy for serverless deployment. This project trains a **small subset** directly in the browser with TensorFlow.js. The Vercel backend only serves small API routes, so deployment is simple.

---

## Features implemented in the web app

- HAM10000 metadata upload
- HAM10000 image/ZIP upload
- Demo subset button
- EDA:
  - metadata count
  - 7-class distribution
  - sample image grid
- Data leakage prevention:
  - split by `lesion_id`, not by image row
  - leakage check between train/validation/test
- Preprocessing:
  - RGB image loading
  - center crop
  - resize to `64x64`
  - normalize to `[0, 1]`
- Train-only augmentation:
  - flip
  - rotate
  - zoom/crop
  - brightness/contrast
- Models:
  - LeNet5 baseline
  - SmallResNet-lite main model
  - linear one-vs-rest SVM on CNN embeddings
- Evaluation:
  - accuracy
  - balanced accuracy
  - macro-F1
  - classification report
  - confusion matrix
  - ROC-AUC for MEL vs non-MEL
  - GroupKFold preview by `lesion_id`
- Downloads:
  - selected subset CSV
  - training history CSV
  - summary JSON
  - checkpoint JSON
  - TFJS model JSON + weights BIN
  - SVM JSON

---

## Local run

```bash
npm install
npm run dev
```

Open:

```text
http://localhost:3000
```

---

## Deploy to Vercel

```bash
npm install -g vercel
vercel
```

Or push this folder to GitHub and import the repository on Vercel. Vercel auto-detects Next.js.

---

## Create a small HAM10000 subset ZIP in Colab

```bash
pip install kagglehub pandas pillow tqdm
python tools/create_subset.py --data-dir auto --per-class 8 --out ham10000_small_subset.zip
```

Then open the web app and upload:

1. `HAM10000_metadata_subset.csv`
2. the image files, or the ZIP containing images

---

## Create a small subset in Kaggle Notebook

Add dataset:

```text
skin-cancer-mnist-ham10000
```

Run:

```bash
python tools/create_subset.py \
  --data-dir /kaggle/input/skin-cancer-mnist-ham10000 \
  --per-class 8 \
  --out ham10000_small_subset.zip
```

---

## Artifact mapping from old pipeline

The old PyTorch pipeline saved `.pth`, `.pt`, and `.joblib`. The Vercel-light version exports web artifacts instead.

| Old artifact | New Vercel-light artifact |
|---|---|
| `models/cnn_smallresnet_multi.pth` | `models/cnn_smallresnet_multi_tfjs_model.json` + `models/cnn_smallresnet_multi_tfjs_weights.bin` |
| `models/cnn_smallresnet_binary.pth` | `models/cnn_smallresnet_binary_tfjs_model.json` + `models/cnn_smallresnet_binary_tfjs_weights.bin` |
| `models/checkpoint_smallresnet_multi.pt` | `models/checkpoint_smallresnet_multi.json` |
| `models/checkpoint_smallresnet_binary.pt` | `models/checkpoint_smallresnet_binary.json` |
| `models/history_smallresnet_multi.csv` | `models/history_smallresnet_multi.csv` |
| `models/history_smallresnet_binary.csv` | `models/history_smallresnet_binary.csv` |
| `models/summary_smallresnet_multi.json` | `models/summary_smallresnet_multi.json` |
| `models/summary_smallresnet_binary.json` | `models/summary_smallresnet_binary.json` |
| `models/svm_embedding_multi.joblib` | `models/svm_embedding_smallresnet_multi.json` |
| `models/svm_embedding_binary.joblib` | `models/svm_embedding_smallresnet_binary.json` |

---

## Important limitation

Vercel is not a GPU training platform. This app is intentionally for **small subset training/demo**. For the full HAM10000 dataset, use Colab/Kaggle/GPU server and keep the previous modular Python pipeline.
