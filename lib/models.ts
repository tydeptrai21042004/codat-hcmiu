import { IMG_SIZE } from "./constants";
import type { ModelKind } from "./types";

function residualBlock(tf: any, input: any, filters: number, name: string): any {
  const conv1 = tf.layers
    .conv2d({ filters, kernelSize: 3, padding: "same", activation: "relu", name: `${name}_conv1` })
    .apply(input);
  const bn1 = tf.layers.batchNormalization({ name: `${name}_bn1` }).apply(conv1);
  const conv2 = tf.layers
    .conv2d({ filters, kernelSize: 3, padding: "same", activation: undefined, name: `${name}_conv2` })
    .apply(bn1);
  const bn2 = tf.layers.batchNormalization({ name: `${name}_bn2` }).apply(conv2);

  let shortcut = input;
  const inputChannels = input.shape[input.shape.length - 1];
  if (inputChannels !== filters) {
    shortcut = tf.layers
      .conv2d({ filters, kernelSize: 1, padding: "same", name: `${name}_shortcut` })
      .apply(input);
  }

  const added = tf.layers.add({ name: `${name}_add` }).apply([shortcut, bn2]);
  return tf.layers.activation({ activation: "relu", name: `${name}_relu` }).apply(added);
}

export function buildLeNet5(tf: any, numClasses: number, learningRate = 0.001): any {
  const model = tf.sequential({ name: "LeNet5_64x64" });
  model.add(tf.layers.conv2d({ inputShape: [IMG_SIZE, IMG_SIZE, 3], filters: 6, kernelSize: 5, activation: "relu", padding: "same" }));
  model.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }));
  model.add(tf.layers.conv2d({ filters: 16, kernelSize: 5, activation: "relu" }));
  model.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }));
  model.add(tf.layers.flatten());
  model.add(tf.layers.dense({ units: 120, activation: "relu" }));
  model.add(tf.layers.dropout({ rate: 0.25 }));
  model.add(tf.layers.dense({ units: 84, activation: "relu", name: "embedding" }));
  model.add(tf.layers.dense({ units: numClasses, activation: "softmax", name: "classifier" }));
  model.compile({ optimizer: tf.train.adam(learningRate), loss: "categoricalCrossentropy", metrics: ["accuracy"] });
  return model;
}

export function buildSmallResNet(tf: any, numClasses: number, learningRate = 0.001): any {
  const input = tf.input({ shape: [IMG_SIZE, IMG_SIZE, 3], name: "image" });
  let x = tf.layers.conv2d({ filters: 16, kernelSize: 3, padding: "same", activation: "relu", name: "stem_conv" }).apply(input);
  x = tf.layers.maxPooling2d({ poolSize: 2, strides: 2, name: "stem_pool" }).apply(x);
  x = residualBlock(tf, x, 16, "block1");
  x = tf.layers.maxPooling2d({ poolSize: 2, strides: 2, name: "pool1" }).apply(x);
  x = residualBlock(tf, x, 32, "block2");
  x = tf.layers.maxPooling2d({ poolSize: 2, strides: 2, name: "pool2" }).apply(x);
  x = residualBlock(tf, x, 64, "block3");
  x = tf.layers.globalAveragePooling2d({ name: "gap" }).apply(x);
  x = tf.layers.dense({ units: 64, activation: "relu", name: "embedding" }).apply(x);
  x = tf.layers.dropout({ rate: 0.3, name: "dropout" }).apply(x);
  const output = tf.layers.dense({ units: numClasses, activation: "softmax", name: "classifier" }).apply(x);
  const model = tf.model({ inputs: input, outputs: output, name: "SmallResNetLite_64x64" });
  model.compile({ optimizer: tf.train.adam(learningRate), loss: "categoricalCrossentropy", metrics: ["accuracy"] });
  return model;
}

export function buildModel(tf: any, kind: ModelKind, numClasses: number, learningRate: number): any {
  return kind === "lenet" ? buildLeNet5(tf, numClasses, learningRate) : buildSmallResNet(tf, numClasses, learningRate);
}

export function embeddingModel(tf: any, model: any): any {
  const layer = model.getLayer("embedding");
  return tf.model({ inputs: model.inputs, outputs: layer.output });
}

export function predictLabels(probabilities: any): number[] {
  return Array.from(probabilities.argMax(-1).dataSync()) as number[];
}
