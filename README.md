# HAM10000 Light ML Pipeline for Vercel

A lightweight full-stack web app for a small HAM10000/ISIC machine-learning pipeline.

## Tech stack

- Frontend: Vite + React + TypeScript
- Backend: Vercel Serverless Functions in `api/`
- ML runtime: TensorFlow.js loaded in the browser by CDN
- Deployment: Vercel

This avoids deploying PyTorch, Flask, scikit-learn, CUDA, or large model files to Vercel.

## Web pipeline features

- Upload a small HAM10000 subset ZIP/CSV/images
- Demo subset for quick testing
- EDA: metadata count, 7-class distribution, sample image grid
- Leakage-safe split by `lesion_id`
- RGB loading, resize to 64x64, normalize to `[0, 1]`
- Train-only augmentation: flip, rotate, zoom/crop, brightness/contrast
- Models: LeNet5 and SmallResNet-lite
- Hybrid ML: linear SVM-style classifier on CNN embeddings
- Metrics: accuracy, balanced accuracy, macro-F1, classification report, confusion matrix, binary ROC-AUC
- Optional GroupKFold preview by `lesion_id`
- Download selected subset CSV and model/report ZIP artifacts

## Local run

```bash
npm ci --no-audit --no-fund
npm run dev
```

Open `http://localhost:5173`.

## Production build

```bash
npm run typecheck
npm run build
```

## Deploy to Vercel

Push the repo to GitHub and import it on Vercel, or run:

```bash
vercel --prod
```

The project pins Node to `20.x`, uses `npm ci`, and forces the public npm registry to avoid npm/Vercel install crashes.

## Create a small HAM10000 subset

On Colab:

```bash
pip install kagglehub pandas pillow tqdm
python tools/create_subset.py --data-dir auto --per-class 8 --out ham10000_small_subset.zip
```

On Kaggle after adding `skin-cancer-mnist-ham10000`:

```bash
python tools/create_subset.py \
  --data-dir /kaggle/input/skin-cancer-mnist-ham10000 \
  --per-class 8 \
  --out ham10000_small_subset.zip
```

Upload that ZIP in the web app.

## API endpoints

- `/api/health`
- `/api/download-template`
