export const IMG_SIZE = 64;

export const HAM_CLASSES = [
  { code: "akiec", label: "Actinic keratoses / intraepithelial carcinoma", short: "AKIEC" },
  { code: "bcc", label: "Basal cell carcinoma", short: "BCC" },
  { code: "bkl", label: "Benign keratosis-like lesions", short: "BKL" },
  { code: "df", label: "Dermatofibroma", short: "DF" },
  { code: "mel", label: "Melanoma", short: "MEL" },
  { code: "nv", label: "Melanocytic nevi", short: "NV" },
  { code: "vasc", label: "Vascular lesions", short: "VASC" }
] as const;

export const CLASS_CODES = HAM_CLASSES.map((item) => item.code);

export const CLASS_LABEL: Record<string, string> = Object.fromEntries(
  HAM_CLASSES.map((item) => [item.code, item.label])
);

export const CLASS_SHORT: Record<string, string> = Object.fromEntries(
  HAM_CLASSES.map((item) => [item.code, item.short])
);

export const DEFAULT_SETTINGS = {
  subsetPerClass: 8,
  epochs: 4,
  batchSize: 8,
  validationRatio: 0.2,
  testRatio: 0.2,
  learningRate: 0.001,
  svmEpochs: 80,
  svmLambda: 0.001,
  groupK: 3
};
