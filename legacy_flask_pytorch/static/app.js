const fileInput = document.getElementById("fileInput");
const btnPredict = document.getElementById("btnPredict");
const previewImg = document.getElementById("previewImg");
const previewHint = document.getElementById("previewHint");
const statusEl = document.getElementById("status");
const binBox = document.getElementById("binBox");
const multiBox = document.getElementById("multiBox");

function setStatus(text, cls="") {
  statusEl.textContent = text;
  statusEl.className = "status " + cls;
}

function formatPct(x) {
  return (x * 100).toFixed(2) + "%";
}

fileInput.addEventListener("change", () => {
  const f = fileInput.files?.[0];
  if (!f) {
    previewImg.style.display = "none";
    previewHint.style.display = "block";
    setStatus("Ready.");
    return;
  }
  const url = URL.createObjectURL(f);
  previewImg.src = url;
  previewImg.style.display = "block";
  previewHint.style.display = "none";
  binBox.textContent = "No result yet.";
  binBox.className = "box muted";
  multiBox.textContent = "No result yet.";
  multiBox.className = "box muted";
  setStatus("Image loaded. Click Predict.");
});

btnPredict.addEventListener("click", async () => {
  const f = fileInput.files?.[0];
  if (!f) {
    setStatus("Please choose an image first.", "warn");
    return;
  }

  btnPredict.disabled = true;
  setStatus("Predicting...", "");

  try {
    const fd = new FormData();
    fd.append("file", f);

    const res = await fetch("/predict", { method: "POST", body: fd });
    const data = await res.json();

    if (!data.ok) {
      throw new Error(data.error || "Prediction failed");
    }

    setStatus(`Done. Device: ${data.device}`, "good");

    // Binary
    if (data.binary && data.binary.available) {
      const p0 = data.binary.prob_benign;
      const p1 = data.binary.prob_malignant;
      const label = data.binary.pred_label;

      binBox.className = "box";
      binBox.textContent =
        `Prediction: ${label}\n` +
        `P(benign)= ${formatPct(p0)}\n` +
        `P(malignant)= ${formatPct(p1)}\n`;
    } else {
      binBox.className = "box muted";
      binBox.textContent = "Binary model not available.";
    }

    // Multi
    if (data.multi && data.multi.available) {
      const label = data.multi.pred_label;
      const topk = data.multi.topk || [];
      let text = `Prediction: ${label}\n\nTop-k:\n`;
      for (const item of topk) {
        text += `- ${item.class}: ${formatPct(item.prob)}\n`;
      }
      multiBox.className = "box";
      multiBox.textContent = text;
    } else {
      multiBox.className = "box muted";
      multiBox.textContent = "Multi-class model not available.";
    }

  } catch (e) {
    setStatus("Error: " + e.message, "bad");
  } finally {
    btnPredict.disabled = false;
  }
});
