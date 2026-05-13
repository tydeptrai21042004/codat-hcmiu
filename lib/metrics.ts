import { CLASS_CODES } from "./constants";
import type { ClassificationMetrics } from "./types";

function safeDiv(a: number, b: number): number {
  return b === 0 ? 0 : a / b;
}

export function confusionMatrix(yTrue: number[], yPred: number[], numClasses: number): number[][] {
  const matrix = Array.from({ length: numClasses }, () => Array.from({ length: numClasses }, () => 0));
  yTrue.forEach((truth, idx) => {
    const pred = yPred[idx];
    if (truth >= 0 && truth < numClasses && pred >= 0 && pred < numClasses) {
      matrix[truth][pred] += 1;
    }
  });
  return matrix;
}

export function rocAucBinary(yTrue01: number[], scores: number[]): number | null {
  const pairs = yTrue01.map((label, idx) => ({ label, score: scores[idx] })).sort((a, b) => b.score - a.score);
  const positives = yTrue01.filter((label) => label === 1).length;
  const negatives = yTrue01.length - positives;
  if (positives === 0 || negatives === 0) return null;

  let tp = 0;
  let fp = 0;
  let prevTpRate = 0;
  let prevFpRate = 0;
  let auc = 0;

  for (const pair of pairs) {
    if (pair.label === 1) tp += 1;
    else fp += 1;
    const tpr = tp / positives;
    const fpr = fp / negatives;
    auc += (fpr - prevFpRate) * (tpr + prevTpRate) / 2;
    prevTpRate = tpr;
    prevFpRate = fpr;
  }
  return auc;
}

export function computeMetrics(yTrue: number[], yPred: number[], probabilities: number[][], numClasses: number, classCodes: string[] = CLASS_CODES.slice(0, numClasses)): ClassificationMetrics {
  const cm = confusionMatrix(yTrue, yPred, numClasses);
  const correct = yTrue.filter((truth, idx) => truth === yPred[idx]).length;
  const report = classCodes.slice(0, numClasses).map((classCode, c) => {
    const tp = cm[c][c];
    const fp = cm.reduce((sum, row, r) => (r === c ? sum : sum + row[c]), 0);
    const fn = cm[c].reduce((sum, value, pred) => (pred === c ? sum : sum + value), 0);
    const support = cm[c].reduce((sum, value) => sum + value, 0);
    const precision = safeDiv(tp, tp + fp);
    const recall = safeDiv(tp, tp + fn);
    const f1 = safeDiv(2 * precision * recall, precision + recall);
    return { classCode, precision, recall, f1, support };
  });

  const balancedAccuracy = report.reduce((sum, row) => sum + row.recall, 0) / numClasses;
  const macroF1 = report.reduce((sum, row) => sum + row.f1, 0) / numClasses;
  const melIndex = classCodes.indexOf("mel");
  const auc = melIndex >= 0 && probabilities.length > 0
    ? rocAucBinary(yTrue.map((label) => (label === melIndex ? 1 : 0)), probabilities.map((row) => row[melIndex] ?? 0))
    : null;

  return {
    accuracy: safeDiv(correct, yTrue.length),
    balancedAccuracy,
    macroF1,
    confusionMatrix: cm,
    report,
    binaryRocAucMel: auc
  };
}

export function formatPercent(value: number | null | undefined): string {
  if (value == null || Number.isNaN(value)) return "N/A";
  return `${(value * 100).toFixed(2)}%`;
}
