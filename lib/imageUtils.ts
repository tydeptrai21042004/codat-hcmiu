import { IMG_SIZE } from "./constants";
import type { ImageRecord } from "./types";

function fileToDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(file);
  });
}

export async function attachImagesFromFiles(records: ImageRecord[], files: File[]): Promise<ImageRecord[]> {
  const fileMap = new Map<string, File>();
  for (const file of files) {
    const clean = file.name.split("/").pop() ?? file.name;
    const id = clean.replace(/\.(jpg|jpeg|png)$/i, "");
    fileMap.set(id, file);
  }

  const attached: ImageRecord[] = [];
  for (const row of records) {
    const file = fileMap.get(row.image_id);
    if (!file && !row.dataUrl) continue;
    attached.push({
      ...row,
      file,
      objectUrl: file ? URL.createObjectURL(file) : row.objectUrl,
      dataUrl: row.dataUrl
    });
  }
  return attached;
}

async function loadImage(record: ImageRecord): Promise<HTMLImageElement> {
  const src = record.dataUrl ?? (record.file ? await fileToDataUrl(record.file) : record.objectUrl);
  if (!src) throw new Error(`Missing image for ${record.image_id}`);
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error(`Cannot load ${record.image_id}`));
    img.src = src;
  });
}

function drawToCanvas(img: HTMLImageElement, augment = false): HTMLCanvasElement {
  const canvas = document.createElement("canvas");
  canvas.width = IMG_SIZE;
  canvas.height = IMG_SIZE;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Canvas is not supported.");

  ctx.save();
  ctx.translate(IMG_SIZE / 2, IMG_SIZE / 2);

  if (augment) {
    const flip = Math.random() > 0.5 ? -1 : 1;
    const angle = (Math.random() * 24 - 12) * (Math.PI / 180);
    const zoom = 1 + Math.random() * 0.12;
    ctx.scale(flip * zoom, zoom);
    ctx.rotate(angle);
  }

  const side = Math.min(img.width, img.height);
  const sx = (img.width - side) / 2;
  const sy = (img.height - side) / 2;
  ctx.drawImage(img, sx, sy, side, side, -IMG_SIZE / 2, -IMG_SIZE / 2, IMG_SIZE, IMG_SIZE);
  ctx.restore();

  if (augment) {
    const imageData = ctx.getImageData(0, 0, IMG_SIZE, IMG_SIZE);
    const brightness = Math.floor(Math.random() * 18 - 9);
    const contrast = 1 + (Math.random() * 0.18 - 0.09);
    for (let i = 0; i < imageData.data.length; i += 4) {
      imageData.data[i] = Math.max(0, Math.min(255, (imageData.data[i] - 128) * contrast + 128 + brightness));
      imageData.data[i + 1] = Math.max(0, Math.min(255, (imageData.data[i + 1] - 128) * contrast + 128 + brightness));
      imageData.data[i + 2] = Math.max(0, Math.min(255, (imageData.data[i + 2] - 128) * contrast + 128 + brightness));
    }
    ctx.putImageData(imageData, 0, 0);
  }

  return canvas;
}

export async function recordToTensor(tf: any, record: ImageRecord, augment = false): Promise<any> {
  const img = await loadImage(record);
  const canvas = drawToCanvas(img, augment);
  return tf.tidy(() => tf.browser.fromPixels(canvas, 3).toFloat().div(255));
}

export async function recordsToTensor4D(tf: any, records: ImageRecord[], augmentTrain = false): Promise<any> {
  const tensors: any[] = [];
  for (const record of records) {
    tensors.push(await recordToTensor(tf, record, augmentTrain));
  }
  const stacked = tf.stack(tensors);
  tensors.forEach((tensor) => tensor.dispose());
  return stacked;
}

export function toOneHot(tf: any, labels: number[], numClasses: number): any {
  return tf.oneHot(tf.tensor1d(labels, "int32"), numClasses);
}
