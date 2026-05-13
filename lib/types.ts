export type DxCode = "akiec" | "bcc" | "bkl" | "df" | "mel" | "nv" | "vasc";
export type SplitName = "train" | "val" | "test";
export type ModelKind = "lenet" | "smallresnet";

export interface HamMetadataRow {
  lesion_id: string;
  image_id: string;
  dx: DxCode;
  dx_type?: string;
  age?: string;
  sex?: string;
  localization?: string;
}

export interface ImageRecord extends HamMetadataRow {
  file?: File;
  objectUrl?: string;
  dataUrl?: string;
  split?: SplitName;
}

export interface ClassCount {
  code: string;
  label: string;
  count: number;
}

export interface SplitResult {
  train: ImageRecord[];
  val: ImageRecord[];
  test: ImageRecord[];
  leakagePairs: string[];
  splitTable: Array<{ split: SplitName; rows: number; lesions: number }>;
}

export interface TensorDataset {
  xs: any;
  ys: any;
  labels: number[];
  records: ImageRecord[];
}

export interface ClassificationMetrics {
  accuracy: number;
  balancedAccuracy: number;
  macroF1: number;
  confusionMatrix: number[][];
  report: Array<{
    classCode: string;
    precision: number;
    recall: number;
    f1: number;
    support: number;
  }>;
  binaryRocAucMel?: number | null;
}

export interface TrainingHistoryRow {
  epoch: number;
  loss: number;
  acc?: number;
  val_loss?: number;
  val_acc?: number;
}

export interface PipelineSummary {
  createdAt: string;
  stack: string;
  imageSize: number;
  classes: string[];
  selectedRows: number;
  split: Array<{ split: SplitName; rows: number; lesions: number }>;
  model: string;
  epochs: number;
  metrics: ClassificationMetrics;
  notes: string[];
}

export interface SvmModelJson {
  classes: string[];
  featureDim: number;
  weights: number[][];
  bias: number[];
  lambda: number;
  epochs: number;
}
