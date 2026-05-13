"use client";

import JSZip from "jszip";
import { useMemo, useState } from "react";
import { CLASS_CODES, CLASS_SHORT, DEFAULT_SETTINGS } from "../lib/constants";
import { parseMetadataCsv, rowsToCsv } from "../lib/csv";
import { createDemoSubset } from "../lib/demoData";
import { downloadPipelineArtifacts } from "../lib/exportArtifacts";
import { attachImagesFromFiles, recordsToTensor4D, toOneHot } from "../lib/imageUtils";
import { computeMetrics, formatPercent } from "../lib/metrics";
import { buildModel, embeddingModel, predictLabels } from "../lib/models";
import { classDistribution, createGroupKFolds, selectBalancedSubset, splitByLesionId } from "../lib/split";
import { predictSvm, softmaxRows, trainLinearSvmOvr } from "../lib/svm";
import type { ClassificationMetrics, ImageRecord, ModelKind, PipelineSummary, SplitResult, SvmModelJson, TrainingHistoryRow } from "../lib/types";

type TaskMode = "multi" | "binary";

function metricText(value?: number | null) {
  return value == null ? "N/A" : formatPercent(value);
}

function getClassCodes(taskMode: TaskMode): string[] {
  return taskMode === "binary" ? ["non_mel", "mel"] : [...CLASS_CODES];
}

function labelIndex(row: ImageRecord, taskMode: TaskMode): number {
  if (taskMode === "binary") return row.dx === "mel" ? 1 : 0;
  return CLASS_CODES.indexOf(row.dx);
}

function fileInputToArray(files: FileList | null): File[] {
  return files ? Array.from(files) : [];
}

declare global {
  interface Window {
    tf?: any;
  }
}

async function loadTf(): Promise<any> {
  if (typeof window === "undefined") throw new Error("TensorFlow.js is available only in the browser.");
  if (window.tf) return window.tf;
  await new Promise<void>((resolve, reject) => {
    const existing = document.querySelector<HTMLScriptElement>("script[data-tfjs]");
    if (existing) {
      existing.addEventListener("load", () => resolve(), { once: true });
      existing.addEventListener("error", () => reject(new Error("Could not load TensorFlow.js CDN script.")), { once: true });
      return;
    }
    const script = document.createElement("script");
    script.src = "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js";
    script.async = true;
    script.dataset.tfjs = "true";
    script.onload = () => resolve();
    script.onerror = () => reject(new Error("Could not load TensorFlow.js CDN script."));
    document.head.appendChild(script);
  });
  if (!window.tf) throw new Error("TensorFlow.js loaded but window.tf is missing.");
  return window.tf;
}

async function expandZipPayload(files: File[]): Promise<{ images: File[]; metadataRows: ImageRecord[] }> {
  const images: File[] = [];
  const metadataRows: ImageRecord[] = [];
  for (const file of files) {
    if (file.name.toLowerCase().endsWith(".zip")) {
      const zip = await JSZip.loadAsync(await file.arrayBuffer());
      const entries = Object.values(zip.files).filter((entry) => !entry.dir);
      for (const entry of entries) {
        const baseName = entry.name.split("/").pop() ?? entry.name;
        if (/\.(jpg|jpeg|png)$/i.test(baseName)) {
          const blob = await entry.async("blob");
          images.push(new File([blob], baseName, { type: blob.type || "image/jpeg" }));
        } else if (/\.csv$/i.test(baseName)) {
          const blob = await entry.async("blob");
          const csvFile = new File([blob], baseName, { type: "text/csv" });
          const rows = await parseMetadataCsv(csvFile);
          metadataRows.push(...(rows as ImageRecord[]));
        }
      }
    } else if (/\.(jpg|jpeg|png)$/i.test(file.name)) {
      images.push(file);
    } else if (/\.csv$/i.test(file.name)) {
      const rows = await parseMetadataCsv(file);
      metadataRows.push(...(rows as ImageRecord[]));
    }
  }
  return { images, metadataRows };
}

export default function Home() {
  const [records, setRecords] = useState<ImageRecord[]>([]);
  const [subset, setSubset] = useState<ImageRecord[]>([]);
  const [split, setSplit] = useState<SplitResult | null>(null);
  const [logs, setLogs] = useState<string[]>([]);
  const [running, setRunning] = useState(false);
  const [modelKind, setModelKind] = useState<ModelKind>("smallresnet");
  const [taskMode, setTaskMode] = useState<TaskMode>("multi");
  const [subsetPerClass, setSubsetPerClass] = useState(DEFAULT_SETTINGS.subsetPerClass);
  const [epochs, setEpochs] = useState(DEFAULT_SETTINGS.epochs);
  const [batchSize, setBatchSize] = useState(DEFAULT_SETTINGS.batchSize);
  const [trainSvm, setTrainSvm] = useState(true);
  const [metrics, setMetrics] = useState<ClassificationMetrics | null>(null);
  const [svmMetrics, setSvmMetrics] = useState<ClassificationMetrics | null>(null);
  const [history, setHistory] = useState<TrainingHistoryRow[]>([]);
  const [trainedModel, setTrainedModel] = useState<any | null>(null);
  const [svmModel, setSvmModel] = useState<SvmModelJson | null>(null);
  const [summary, setSummary] = useState<PipelineSummary | null>(null);
  const [foldText, setFoldText] = useState<string>("");

  const distribution = useMemo(() => classDistribution(subset.length ? subset : records), [records, subset]);
  const maxCount = Math.max(1, ...distribution.map((item) => item.count));
  const sampleImages = useMemo(() => {
    const source = subset.length ? subset : records;
    return CLASS_CODES.map((code) => source.find((row) => row.dx === code)).filter(Boolean) as ImageRecord[];
  }, [records, subset]);

  function log(message: string) {
    setLogs((prev) => [...prev, `[${new Date().toLocaleTimeString()}] ${message}`]);
  }

  async function loadDemo() {
    const demo = createDemoSubset(Math.max(3, subsetPerClass));
    setRecords(demo);
    const selected = selectBalancedSubset(demo, subsetPerClass);
    const splitResult = splitByLesionId(selected, DEFAULT_SETTINGS.validationRatio, DEFAULT_SETTINGS.testRatio);
    setSubset(selected);
    setSplit(splitResult);
    setMetrics(null);
    setSvmMetrics(null);
    log(`Loaded demo subset: ${selected.length} images. This is synthetic data for UI/testing only.`);
  }

  async function handleMetadataUpload(files: FileList | null) {
    const file = fileInputToArray(files)[0];
    if (!file) return;
    const rows = await parseMetadataCsv(file);
    setRecords(rows as ImageRecord[]);
    setSubset([]);
    setSplit(null);
    log(`Loaded metadata CSV: ${rows.length} valid HAM10000 rows.`);
  }

  async function handleImageUpload(files: FileList | null) {
    const selected = fileInputToArray(files);
    if (!selected.length) return;
    setRunning(true);
    try {
      const payload = await expandZipPayload(selected);
      const baseRecords = payload.metadataRows.length ? payload.metadataRows : records;
      if (payload.metadataRows.length) {
        setRecords(payload.metadataRows);
        log(`Found metadata CSV inside upload: ${payload.metadataRows.length} valid rows.`);
      }
      const attached = await attachImagesFromFiles(baseRecords, payload.images);
      const balanced = selectBalancedSubset(attached, subsetPerClass);
      const splitResult = splitByLesionId(balanced, DEFAULT_SETTINGS.validationRatio, DEFAULT_SETTINGS.testRatio);
      setSubset(balanced);
      setSplit(splitResult);
      log(`Matched ${attached.length} uploaded images to metadata; selected balanced subset of ${balanced.length}.`);
      if (splitResult.leakagePairs.length === 0) log("Leakage check passed: no lesion_id appears in more than one split.");
      else log(`Leakage warning: ${splitResult.leakagePairs.join(", ")}`);
    } catch (err) {
      log(`Image loading failed: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setRunning(false);
    }
  }

  function makeSplitAgain() {
    const source = subset.length ? subset : selectBalancedSubset(records, subsetPerClass);
    const splitResult = splitByLesionId(source, DEFAULT_SETTINGS.validationRatio, DEFAULT_SETTINGS.testRatio, Math.floor(Math.random() * 10000));
    setSubset(source);
    setSplit(splitResult);
    log(`Created lesion_id split: train=${splitResult.train.length}, val=${splitResult.val.length}, test=${splitResult.test.length}.`);
  }

  function previewGroupKFold() {
    const source = subset.length ? subset : records;
    const folds = createGroupKFolds(source, DEFAULT_SETTINGS.groupK);
    const text = folds
      .map((fold) => `Fold ${fold.fold}: train=${fold.train.length}, val=${fold.val.length}, leakage=${fold.leakage.length}`)
      .join("\n");
    setFoldText(text);
    log("Generated GroupKFold preview by lesion_id.");
  }

  async function trainPipeline() {
    if (!split) {
      log("Please load data and create a lesion_id split first.");
      return;
    }
    const classes = getClassCodes(taskMode);
    const numClasses = classes.length;
    if (split.train.length < numClasses || split.test.length === 0) {
      log("Subset is too small for training/evaluation. Increase subset per class or upload more images.");
      return;
    }

    setRunning(true);
    setMetrics(null);
    setSvmMetrics(null);
    setHistory([]);
    setTrainedModel(null);
    setSvmModel(null);
    setSummary(null);

    let trainXs: any | null = null;
    let trainAugXs: any | null = null;
    let valXs: any | null = null;
    let testXs: any | null = null;
    let trainYs: any | null = null;
    let trainAugYs: any | null = null;
    let valYs: any | null = null;

    try {
      const tf = await loadTf();
      await tf.setBackend("webgl").catch(async () => tf.setBackend("cpu"));
      await tf.ready();
      log(`TensorFlow.js backend: ${tf.getBackend()}`);
      log("Preprocessing RGB images: center crop, resize 64x64, normalize to [0, 1].");

      const trainLabels = split.train.map((row) => labelIndex(row, taskMode));
      const valLabels = split.val.map((row) => labelIndex(row, taskMode));
      const testLabels = split.test.map((row) => labelIndex(row, taskMode));

      trainXs = await recordsToTensor4D(tf, split.train, false);
      trainAugXs = await recordsToTensor4D(tf, split.train, true);
      valXs = await recordsToTensor4D(tf, split.val, false);
      testXs = await recordsToTensor4D(tf, split.test, false);
      trainYs = toOneHot(tf, trainLabels, numClasses);
      trainAugYs = toOneHot(tf, [...trainLabels, ...trainLabels], numClasses);
      valYs = toOneHot(tf, valLabels, numClasses);

      const trainAllXs = tf.concat([trainXs, trainAugXs], 0) as any;
      const model = buildModel(tf, modelKind, numClasses, DEFAULT_SETTINGS.learningRate);
      log(`Training ${model.name} with train-only augmentation for ${epochs} epochs.`);

      const histRows: TrainingHistoryRow[] = [];
      await model.fit(trainAllXs, trainAugYs, {
        epochs,
        batchSize,
        shuffle: true,
        validationData: [valXs, valYs],
        callbacks: {
          onEpochEnd: async (epoch: number, logsObj: Record<string, number> | undefined) => {
            const row: TrainingHistoryRow = {
              epoch: epoch + 1,
              loss: Number(logsObj?.loss ?? 0),
              acc: Number(logsObj?.acc ?? logsObj?.accuracy ?? 0),
              val_loss: Number(logsObj?.val_loss ?? 0),
              val_acc: Number(logsObj?.val_acc ?? logsObj?.val_accuracy ?? 0)
            };
            histRows.push(row);
            setHistory([...histRows]);
            log(`Epoch ${row.epoch}/${epochs}: loss=${row.loss.toFixed(4)}, val_acc=${((row.val_acc ?? 0) * 100).toFixed(2)}%`);
            await tf.nextFrame();
          }
        }
      });
      trainAllXs.dispose();

      const probTensor = model.predict(testXs) as any;
      const probabilities = (await probTensor.array()) as number[][];
      const pred = predictLabels(probTensor);
      probTensor.dispose();
      const cnnMetrics = computeMetrics(testLabels, pred, probabilities, numClasses, classes);
      setMetrics(cnnMetrics);
      setTrainedModel(model);
      setHistory(histRows);
      log(`CNN evaluation done: acc=${formatPercent(cnnMetrics.accuracy)}, macro-F1=${formatPercent(cnnMetrics.macroF1)}.`);

      let svm: SvmModelJson | null = null;
      let svmEval: ClassificationMetrics | null = null;
      if (trainSvm) {
        log("Extracting CNN embeddings and training linear one-vs-rest SVM in browser.");
        const emb = embeddingModel(tf, model);
        const trainFeatTensor = emb.predict(trainXs) as any;
        const testFeatTensor = emb.predict(testXs) as any;
        const trainFeatures = (await trainFeatTensor.array()) as number[][];
        const testFeatures = (await testFeatTensor.array()) as number[][];
        trainFeatTensor.dispose();
        testFeatTensor.dispose();
        emb.dispose();
        svm = trainLinearSvmOvr(trainFeatures, trainLabels, numClasses, DEFAULT_SETTINGS.svmEpochs, DEFAULT_SETTINGS.svmLambda);
        const svmPred = predictSvm(svm, testFeatures);
        svmEval = computeMetrics(testLabels, svmPred.labels, softmaxRows(svmPred.scores), numClasses, classes);
        setSvmModel(svm);
        setSvmMetrics(svmEval);
        log(`SVM evaluation done: acc=${formatPercent(svmEval.accuracy)}, macro-F1=${formatPercent(svmEval.macroF1)}.`);
      }

      const pipelineSummary: PipelineSummary = {
        createdAt: new Date().toISOString(),
        stack: "Next.js + TypeScript + TensorFlow.js; ML training runs client-side for Vercel-friendly deployment.",
        imageSize: 64,
        classes,
        selectedRows: subset.length,
        split: split.splitTable,
        model: model.name,
        epochs,
        metrics: cnnMetrics,
        notes: [
          "HAM10000 split is grouped by lesion_id to avoid train/validation/test leakage.",
          "Augmentation is applied only to train images.",
          "This browser version exports TFJS model files instead of PyTorch .pth files.",
          svmEval ? "SVM on CNN embeddings was trained and evaluated." : "SVM step was skipped."
        ]
      };
      setSummary(pipelineSummary);
    } catch (err) {
      log(`Training failed: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      trainXs?.dispose();
      trainAugXs?.dispose();
      valXs?.dispose();
      testXs?.dispose();
      trainYs?.dispose();
      trainAugYs?.dispose();
      valYs?.dispose();
      setRunning(false);
    }
  }

  async function downloadArtifacts() {
    if (!summary) {
      log("Train a model first, then download artifacts.");
      return;
    }
    await downloadPipelineArtifacts({
      tf: await loadTf(),
      model: trainedModel,
      modelName: `ham10000_${taskMode}_${modelKind}`,
      filePrefix: taskMode === "binary" ? "smallresnet_binary" : "smallresnet_multi",
      history,
      summary,
      svm: svmModel
    });
    log("Downloaded ZIP artifacts.");
  }

  function downloadSubsetCsv() {
    const rows = (subset.length ? subset : records).map(({ objectUrl, dataUrl, file, ...row }) => row);
    const blob = new Blob([rowsToCsv(rows as unknown as Array<Record<string, unknown>>)], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "selected_ham10000_subset_metadata.csv";
    a.click();
    URL.revokeObjectURL(url);
  }

  return (
    <main className="container">
      <section className="hero">
        <h1>HAM10000 Light ML Pipeline for Vercel</h1>
        <p>
          A browser-first machine learning pipeline: upload a small HAM10000 subset, inspect EDA, split safely by lesion_id,
          preprocess 64×64 RGB images, train LeNet5 or SmallResNet-lite, evaluate metrics, train SVM on CNN embeddings,
          then download model and reports.
        </p>
        <div className="badges">
          <span className="badge">Next.js FE + backend</span>
          <span className="badge">TensorFlow.js browser training</span>
          <span className="badge">No Flask server needed</span>
          <span className="badge">Vercel-ready</span>
        </div>
      </section>

      <section className="grid">
        <div className="card two">
          <h2>1. Load HAM10000 subset</h2>
          <p>Use the helper script in <code>tools/create_subset.py</code> on Colab/Kaggle to create a tiny ZIP, or upload metadata/images manually.</p>
          <div className="controls">
            <div className="field">
              <label>Metadata CSV</label>
              <input type="file" accept=".csv" onChange={(e) => handleMetadataUpload(e.target.files)} />
            </div>
            <div className="field">
              <label>Images or ZIP</label>
              <input type="file" accept=".jpg,.jpeg,.png,.zip,.csv" multiple onChange={(e) => handleImageUpload(e.target.files)} />
            </div>
          </div>
          <div className="controls" style={{ marginTop: "0.9rem" }}>
            <button className="secondary" onClick={loadDemo} disabled={running}>Use demo subset</button>
            <a href="/api/download-template">Download CSV template</a>
          </div>
        </div>

        <div className="card two">
          <h2>2. Configure small run</h2>
          <div className="controls">
            <div className="field">
              <label>Subset / class</label>
              <input type="number" min={2} max={30} value={subsetPerClass} onChange={(e) => setSubsetPerClass(Number(e.target.value))} />
            </div>
            <div className="field">
              <label>Model</label>
              <select value={modelKind} onChange={(e) => setModelKind(e.target.value as ModelKind)}>
                <option value="lenet">LeNet5 baseline</option>
                <option value="smallresnet">SmallResNet-lite</option>
              </select>
            </div>
            <div className="field">
              <label>Task</label>
              <select value={taskMode} onChange={(e) => setTaskMode(e.target.value as TaskMode)}>
                <option value="multi">7-class multi-class</option>
                <option value="binary">Binary MEL vs non-MEL</option>
              </select>
            </div>
            <div className="field">
              <label>Epochs</label>
              <input type="number" min={1} max={20} value={epochs} onChange={(e) => setEpochs(Number(e.target.value))} />
            </div>
            <div className="field">
              <label>Batch</label>
              <input type="number" min={2} max={32} value={batchSize} onChange={(e) => setBatchSize(Number(e.target.value))} />
            </div>
            <label style={{ display: "flex", gap: "0.45rem", alignItems: "center", fontWeight: 800 }}>
              <input type="checkbox" checked={trainSvm} onChange={(e) => setTrainSvm(e.target.checked)} /> Train SVM embeddings
            </label>
          </div>
          <div className="controls" style={{ marginTop: "0.9rem" }}>
            <button className="ghost" onClick={makeSplitAgain} disabled={running || records.length === 0}>Create lesion_id split</button>
            <button className="ghost" onClick={previewGroupKFold} disabled={running || (subset.length || records.length) === 0}>Preview GroupKFold</button>
            <button onClick={trainPipeline} disabled={running || !split}>Run web pipeline</button>
          </div>
        </div>

        <div className="card two">
          <h2>3. EDA: class distribution</h2>
          <div className="table-wrap">
            <table>
              <thead><tr><th>Class</th><th>Count</th><th>Bar</th></tr></thead>
              <tbody>
                {distribution.map((row) => (
                  <tr key={row.code}>
                    <td><strong>{CLASS_SHORT[row.code] ?? row.code}</strong><br /><span className="muted">{row.label}</span></td>
                    <td>{row.count}</td>
                    <td><div className="bar"><div style={{ width: `${(row.count / maxCount) * 100}%` }} /></div></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <div className="card two">
          <h2>4. Sample images</h2>
          {sampleImages.length === 0 ? <p className="muted">No images loaded yet.</p> : (
            <div className="image-grid">
              {sampleImages.map((row) => (
                <div className="sample" key={row.image_id}>
                  <img src={row.dataUrl ?? row.objectUrl} alt={row.image_id} />
                  <div>{CLASS_SHORT[row.dx]} · {row.image_id}</div>
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="card two">
          <h2>5. Leakage-safe split</h2>
          {!split ? <p className="muted">Create a split after loading data.</p> : (
            <>
              <div className="table-wrap">
                <table>
                  <thead><tr><th>Split</th><th>Rows</th><th>Unique lesion_id</th></tr></thead>
                  <tbody>{split.splitTable.map((row) => <tr key={row.split}><td>{row.split}</td><td>{row.rows}</td><td>{row.lesions}</td></tr>)}</tbody>
                </table>
              </div>
              <p className={split.leakagePairs.length ? "status-bad" : "status-ok"}>
                {split.leakagePairs.length ? `Leakage detected: ${split.leakagePairs.join(", ")}` : "Leakage check passed: lesion_id groups do not overlap."}
              </p>
            </>
          )}
          {foldText && <pre className="log">{foldText}</pre>}
        </div>

        <div className="card two">
          <h2>6. Evaluation</h2>
          {!metrics ? <p className="muted">Run training to see metrics.</p> : (
            <div className="grid" style={{ marginTop: 0 }}>
              <div className="metric three"><span>Accuracy</span><strong>{metricText(metrics.accuracy)}</strong></div>
              <div className="metric three"><span>Balanced acc.</span><strong>{metricText(metrics.balancedAccuracy)}</strong></div>
              <div className="metric three"><span>Macro-F1</span><strong>{metricText(metrics.macroF1)}</strong></div>
              <div className="metric three"><span>ROC-AUC MEL</span><strong>{metricText(metrics.binaryRocAucMel)}</strong></div>
            </div>
          )}
          {svmMetrics && <p><strong>SVM embeddings:</strong> acc={metricText(svmMetrics.accuracy)}, macro-F1={metricText(svmMetrics.macroF1)}</p>}
        </div>

        {metrics && (
          <div className="card">
            <h2>Classification report</h2>
            <div className="table-wrap">
              <table>
                <thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr></thead>
                <tbody>{metrics.report.map((row) => <tr key={row.classCode}><td>{row.classCode}</td><td>{row.precision.toFixed(3)}</td><td>{row.recall.toFixed(3)}</td><td>{row.f1.toFixed(3)}</td><td>{row.support}</td></tr>)}</tbody>
              </table>
            </div>
          </div>
        )}

        {metrics && (
          <div className="card">
            <h2>Confusion matrix</h2>
            <div className="table-wrap">
              <table>
                <tbody>{metrics.confusionMatrix.map((row, idx) => <tr key={idx}><th>{getClassCodes(taskMode)[idx]}</th>{row.map((value, j) => <td key={j}>{value}</td>)}</tr>)}</tbody>
              </table>
            </div>
          </div>
        )}

        <div className="card">
          <h2>7. Download outputs</h2>
          <p>
            Download selected subset metadata, training history, summary JSON, checkpoint JSON, TFJS model files, and SVM JSON.
            This replaces heavy PyTorch/Flask artifacts with deployable web artifacts.
          </p>
          <div className="controls">
            <button className="ghost" onClick={downloadSubsetCsv} disabled={(subset.length || records.length) === 0}>Download subset CSV</button>
            <button onClick={downloadArtifacts} disabled={!summary}>Download model + reports ZIP</button>
          </div>
        </div>

        <div className="card">
          <h2>Run log</h2>
          <pre className="log">{logs.length ? logs.join("\n") : "No logs yet."}</pre>
        </div>
      </section>

      <p className="footer">
        Research/education use only. This demo is not a medical diagnostic system. For full HAM10000 training, use Colab/Kaggle or a GPU server; for Vercel, use this small subset browser pipeline.
      </p>
    </main>
  );
}
