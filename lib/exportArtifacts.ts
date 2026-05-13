import JSZip from "jszip";
import { saveAs } from "file-saver";
import { rowsToCsv } from "./csv";
import type { PipelineSummary, SvmModelJson, TrainingHistoryRow } from "./types";

function tensorDataToArrayBuffer(data: any): ArrayBuffer {
  if (Array.isArray(data)) {
    const total = data.reduce((sum: number, item: ArrayBuffer) => sum + item.byteLength, 0);
    const out = new Uint8Array(total);
    let offset = 0;
    for (const item of data as ArrayBuffer[]) {
      out.set(new Uint8Array(item), offset);
      offset += item.byteLength;
    }
    return out.buffer;
  }
  return data as ArrayBuffer;
}

async function addTfjsModel(tf: any, zip: JSZip, folderName: string, model: any): Promise<void> {
  let captured: any = null;
  await model.save(
    tf.io.withSaveHandler(async (modelArtifacts: any) => {
      captured = modelArtifacts;
      return {
        modelArtifactsInfo: {
          dateSaved: new Date(),
          modelTopologyType: "JSON",
          modelTopologyBytes: JSON.stringify(modelArtifacts.modelTopology ?? {}).length,
          weightSpecsBytes: JSON.stringify(modelArtifacts.weightSpecs ?? []).length,
          weightDataBytes: modelArtifacts.weightData ? tensorDataToArrayBuffer(modelArtifacts.weightData).byteLength : 0
        }
      };
    })
  );
  if (!captured) throw new Error("Unable to capture TensorFlow.js model artifacts.");
  const weightData = captured.weightData ? tensorDataToArrayBuffer(captured.weightData) : new ArrayBuffer(0);
  const modelJson = {
    modelTopology: captured.modelTopology,
    format: captured.format ?? "layers-model",
    generatedBy: captured.generatedBy ?? "TensorFlow.js",
    convertedBy: captured.convertedBy ?? null,
    weightsManifest: [
      {
        paths: [`${folderName}_weights.bin`],
        weights: captured.weightSpecs ?? []
      }
    ]
  };
  zip.file(`models/${folderName}_model.json`, JSON.stringify(modelJson, null, 2));
  zip.file(`models/${folderName}_weights.bin`, weightData);
}

export async function downloadPipelineArtifacts(params: {
  tf: any;
  model?: any | null;
  modelName: string;
  filePrefix?: string;
  history: TrainingHistoryRow[];
  summary: PipelineSummary;
  svm?: SvmModelJson | null;
}): Promise<void> {
  const zip = new JSZip();
  const prefix = params.filePrefix ?? "smallresnet_multi";
  zip.file(`models/summary_${prefix}.json`, JSON.stringify(params.summary, null, 2));
  zip.file(`models/history_${prefix}.csv`, rowsToCsv(params.history as unknown as Array<Record<string, unknown>>));
  zip.file(
    `models/checkpoint_${prefix}.json`,
    JSON.stringify(
      {
        modelName: params.modelName,
        stack: "Next.js + TensorFlow.js browser training",
        createdAt: new Date().toISOString(),
        note: "This Vercel-light version exports TensorFlow.js artifacts instead of PyTorch .pth files."
      },
      null,
      2
    )
  );

  if (params.model) {
    await addTfjsModel(params.tf, zip, `cnn_${prefix}_tfjs`, params.model);
  }

  if (params.svm) {
    zip.file(`models/svm_embedding_${prefix}.json`, JSON.stringify(params.svm, null, 2));
  }

  zip.file(
    "MODEL_NAME_MAPPING.md",
    [
      "# Artifact mapping for the Vercel-light stack",
      "",
      "The original Flask/PyTorch version saved `.pth` and `.joblib` files.",
      "This Vercel-ready version trains in the browser with TensorFlow.js, so the equivalent deployable artifacts are:",
      "",
      "| Original request | Web/Vercel artifact |",
      "|---|---|",
      "| `models/cnn_smallresnet_multi.pth` | `models/cnn_smallresnet_multi_tfjs_model.json` + `models/cnn_smallresnet_multi_tfjs_weights.bin` |",
      "| `models/cnn_smallresnet_binary.pth` | `models/cnn_smallresnet_binary_tfjs_model.json` + `models/cnn_smallresnet_binary_tfjs_weights.bin` |",
      "| `models/checkpoint_smallresnet_multi.pt` | `models/checkpoint_smallresnet_multi.json` |",
      "| `models/checkpoint_smallresnet_binary.pt` | `models/checkpoint_smallresnet_binary.json` |",
      "| `models/history_smallresnet_multi.csv` | `models/history_smallresnet_multi.csv` |",
      "| `models/history_smallresnet_binary.csv` | `models/history_smallresnet_binary.csv` |",
      "| `models/summary_smallresnet_multi.json` | `models/summary_smallresnet_multi.json` |",
      "| `models/summary_smallresnet_binary.json` | `models/summary_smallresnet_binary.json` |",
      "| `models/svm_embedding_multi.joblib` | `models/svm_embedding_smallresnet_multi.json` |",
      "| `models/svm_embedding_binary.joblib` | `models/svm_embedding_smallresnet_binary.json` |"
    ].join("\n")
  );

  const blob = await zip.generateAsync({ type: "blob" });
  saveAs(blob, `${params.modelName}_web_artifacts.zip`);
}
