import { CLASS_CODES } from "./constants";
import type { SvmModelJson } from "./types";

function dot(a: number[], b: number[]): number {
  let s = 0;
  for (let i = 0; i < a.length; i += 1) s += a[i] * b[i];
  return s;
}

export function trainLinearSvmOvr(
  features: number[][],
  labels: number[],
  numClasses: number,
  epochs = 80,
  lambda = 0.001
): SvmModelJson {
  if (features.length === 0) throw new Error("No features for SVM.");
  const dim = features[0].length;
  const weights = Array.from({ length: numClasses }, () => Array.from({ length: dim }, () => 0));
  const bias = Array.from({ length: numClasses }, () => 0);
  let t = 1;

  for (let epoch = 0; epoch < epochs; epoch += 1) {
    for (let i = 0; i < features.length; i += 1) {
      const x = features[i];
      const trueLabel = labels[i];
      for (let c = 0; c < numClasses; c += 1) {
        const y = trueLabel === c ? 1 : -1;
        const eta = 1 / (lambda * t);
        const score = dot(weights[c], x) + bias[c];
        for (let j = 0; j < dim; j += 1) {
          weights[c][j] *= 1 - eta * lambda;
        }
        if (y * score < 1) {
          for (let j = 0; j < dim; j += 1) {
            weights[c][j] += eta * y * x[j];
          }
          bias[c] += eta * y;
        }
        t += 1;
      }
    }
  }

  return {
    classes: CLASS_CODES.slice(0, numClasses),
    featureDim: dim,
    weights,
    bias,
    lambda,
    epochs
  };
}

export function predictSvm(model: SvmModelJson, features: number[][]): { labels: number[]; scores: number[][] } {
  const scores = features.map((x) => model.weights.map((w, c) => dot(w, x) + model.bias[c]));
  const labels = scores.map((row) => row.reduce((best, value, idx) => (value > row[best] ? idx : best), 0));
  return { labels, scores };
}

export function softmaxRows(scores: number[][]): number[][] {
  return scores.map((row) => {
    const max = Math.max(...row);
    const exp = row.map((v) => Math.exp(v - max));
    const sum = exp.reduce((a, b) => a + b, 0) || 1;
    return exp.map((v) => v / sum);
  });
}
