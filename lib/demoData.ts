import { CLASS_CODES, CLASS_SHORT, IMG_SIZE } from "./constants";
import type { DxCode, ImageRecord } from "./types";

function drawSyntheticLesion(classCode: string, idx: number): string {
  const canvas = document.createElement("canvas");
  canvas.width = IMG_SIZE;
  canvas.height = IMG_SIZE;
  const ctx = canvas.getContext("2d");
  if (!ctx) return "";

  const bg = 210 + ((idx * 7) % 25);
  ctx.fillStyle = `rgb(${bg + 20}, ${bg - 5}, ${bg - 18})`;
  ctx.fillRect(0, 0, IMG_SIZE, IMG_SIZE);

  for (let y = 0; y < IMG_SIZE; y += 1) {
    for (let x = 0; x < IMG_SIZE; x += 1) {
      const noise = Math.floor(((x * 17 + y * 13 + idx * 29) % 19) - 9);
      const px = ctx.getImageData(x, y, 1, 1);
      px.data[0] = Math.max(0, Math.min(255, px.data[0] + noise));
      px.data[1] = Math.max(0, Math.min(255, px.data[1] + noise));
      px.data[2] = Math.max(0, Math.min(255, px.data[2] + noise));
      ctx.putImageData(px, x, y);
    }
  }

  const centerX = 28 + ((idx * 3) % 12);
  const centerY = 30 + ((idx * 5) % 10);
  const rx = 8 + ((idx + classCode.length) % 8);
  const ry = 6 + ((idx * 2 + classCode.length) % 8);
  const hueShift = CLASS_CODES.indexOf(classCode as DxCode) * 18;
  ctx.save();
  ctx.translate(centerX, centerY);
  ctx.rotate(((idx % 9) - 4) * 0.08);
  ctx.beginPath();
  ctx.ellipse(0, 0, rx, ry, 0, 0, Math.PI * 2);
  ctx.fillStyle = `rgb(${85 + hueShift}, ${42 + (idx % 20)}, ${35 + hueShift / 2})`;
  ctx.fill();
  ctx.globalAlpha = 0.35;
  ctx.beginPath();
  ctx.ellipse(-2, -2, Math.max(2, rx / 3), Math.max(2, ry / 3), 0, 0, Math.PI * 2);
  ctx.fillStyle = "rgba(255,255,255,0.35)";
  ctx.fill();
  ctx.restore();

  ctx.fillStyle = "rgba(0,0,0,0.55)";
  ctx.font = "10px sans-serif";
  ctx.fillText(CLASS_SHORT[classCode], 4, 12);
  return canvas.toDataURL("image/png");
}

export function createDemoSubset(perClass = 6): ImageRecord[] {
  const rows: ImageRecord[] = [];
  for (const code of CLASS_CODES) {
    for (let i = 0; i < perClass; i += 1) {
      const imageId = `demo_${code}_${String(i).padStart(3, "0")}`;
      rows.push({
        lesion_id: `demo_lesion_${code}_${Math.floor(i / 2)}`,
        image_id: imageId,
        dx: code as DxCode,
        dx_type: "demo",
        age: String(40 + i),
        sex: i % 2 === 0 ? "male" : "female",
        localization: "demo",
        dataUrl: drawSyntheticLesion(code, i)
      });
    }
  }
  return rows;
}
