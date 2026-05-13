import { CLASS_CODES, CLASS_LABEL } from "./constants";
import type { ClassCount, HamMetadataRow, ImageRecord, SplitName, SplitResult } from "./types";

function seededRandom(seed: number): () => number {
  let value = seed % 2147483647;
  if (value <= 0) value += 2147483646;
  return () => {
    value = (value * 16807) % 2147483647;
    return (value - 1) / 2147483646;
  };
}

export function shuffle<T>(items: T[], seed = 42): T[] {
  const arr = [...items];
  const rand = seededRandom(seed);
  for (let i = arr.length - 1; i > 0; i -= 1) {
    const j = Math.floor(rand() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

export function classDistribution(rows: Pick<HamMetadataRow, "dx">[]): ClassCount[] {
  return CLASS_CODES.map((code) => ({
    code,
    label: CLASS_LABEL[code],
    count: rows.filter((row) => row.dx === code).length
  }));
}

export function selectBalancedSubset(records: ImageRecord[], perClass: number, seed = 42): ImageRecord[] {
  const selected: ImageRecord[] = [];
  for (const code of CLASS_CODES) {
    const rows = records.filter((row) => row.dx === code);
    selected.push(...shuffle(rows, seed + code.length).slice(0, perClass));
  }
  return shuffle(selected, seed + 100);
}

function groupByLesion(records: ImageRecord[]): Map<string, ImageRecord[]> {
  const map = new Map<string, ImageRecord[]>();
  for (const row of records) {
    const group = map.get(row.lesion_id) ?? [];
    group.push(row);
    map.set(row.lesion_id, group);
  }
  return map;
}

function lesionSet(records: ImageRecord[]): Set<string> {
  return new Set(records.map((row) => row.lesion_id));
}

function intersection(a: Set<string>, b: Set<string>): string[] {
  return [...a].filter((item) => b.has(item));
}

export function splitByLesionId(
  records: ImageRecord[],
  validationRatio = 0.2,
  testRatio = 0.2,
  seed = 42
): SplitResult {
  const groups = [...groupByLesion(records).entries()].map(([lesionId, rows]) => ({ lesionId, rows }));
  const shuffled = shuffle(groups, seed);
  const n = shuffled.length;
  const testN = Math.max(1, Math.round(n * testRatio));
  const valN = Math.max(1, Math.round(n * validationRatio));

  const testGroups = shuffled.slice(0, testN);
  const valGroups = shuffled.slice(testN, testN + valN);
  const trainGroups = shuffled.slice(testN + valN);

  const train = trainGroups.flatMap((group) => group.rows).map((r) => ({ ...r, split: "train" as SplitName }));
  const val = valGroups.flatMap((group) => group.rows).map((r) => ({ ...r, split: "val" as SplitName }));
  const test = testGroups.flatMap((group) => group.rows).map((r) => ({ ...r, split: "test" as SplitName }));

  const trainLesions = lesionSet(train);
  const valLesions = lesionSet(val);
  const testLesions = lesionSet(test);
  const leakagePairs = [
    ...intersection(trainLesions, valLesions).map((id) => `train-val:${id}`),
    ...intersection(trainLesions, testLesions).map((id) => `train-test:${id}`),
    ...intersection(valLesions, testLesions).map((id) => `val-test:${id}`)
  ];

  return {
    train,
    val,
    test,
    leakagePairs,
    splitTable: [
      { split: "train", rows: train.length, lesions: trainLesions.size },
      { split: "val", rows: val.length, lesions: valLesions.size },
      { split: "test", rows: test.length, lesions: testLesions.size }
    ]
  };
}

export function createGroupKFolds(records: ImageRecord[], k = 3, seed = 42): Array<{ fold: number; train: ImageRecord[]; val: ImageRecord[]; leakage: string[] }> {
  const groups = shuffle([...groupByLesion(records).entries()], seed).map(([lesionId, rows]) => ({ lesionId, rows }));
  const folds: Array<{ fold: number; train: ImageRecord[]; val: ImageRecord[]; leakage: string[] }> = [];
  for (let fold = 0; fold < k; fold += 1) {
    const valGroups = groups.filter((_, idx) => idx % k === fold);
    const trainGroups = groups.filter((_, idx) => idx % k !== fold);
    const train = trainGroups.flatMap((group) => group.rows);
    const val = valGroups.flatMap((group) => group.rows);
    const leakage = intersection(lesionSet(train), lesionSet(val));
    folds.push({ fold: fold + 1, train, val, leakage });
  }
  return folds;
}
