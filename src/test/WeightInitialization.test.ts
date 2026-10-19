import { test } from "node:test";
import assert from "node:assert/strict";
import Hyperparameters from "../Hyperparameters.js";
import GPTModel from "../GPTModel.js";
import createSeededRandom from "../SeededRandom.js";
import { randomNormal, randomNormalMatrix, residualProjectionStandardDeviation } from "../WeightInitialization.js";

/** The mean and standard deviation of every value in a vector or matrix. */
const statistics = (values: number[] | number[][]): { mean: number; standardDeviation: number } => {
  const flat: number[] = values.flat();
  const mean: number = flat.reduce((sum, value) => sum + value, 0) / flat.length;
  const variance: number = flat.reduce((sum, value) => sum + (value - mean) ** 2, 0) / flat.length;
  return { mean, standardDeviation: Math.sqrt(variance) };
};

/*
 * Checks that values look like samples from a normal distribution with mean
 * 0 and the expected standard deviation, within 5% for the standard
 * deviation. The matrices tested have thousands of values, so a correct
 * initialization is far inside these limits.
 */
const assertNormal = (values: number[][], expectedStandardDeviation: number, label: string): void => {
  const { mean, standardDeviation } = statistics(values);
  assert.ok(Math.abs(mean) < 0.1 * expectedStandardDeviation, `${label}: mean ${mean}`);
  assert.ok(
    Math.abs(standardDeviation / expectedStandardDeviation - 1) < 0.05,
    `${label}: standard deviation ${standardDeviation}, expected ${expectedStandardDeviation}`
  );
};

const assertAllZero = (values: number[], label: string): void => {
  assert.ok(values.length > 0, `${label}: empty`);
  assert.ok(values.every((value) => value === 0), `${label}: not all zero`);
};

test("randomNormal() samples have mean 0 and the requested standard deviation", () => {
  const random = createSeededRandom(1);
  const samples = Array.from({ length: 20000 }, () => randomNormal(random, 0.02));

  assertNormal([samples], 0.02, "samples");
});

test("randomNormalMatrix() has the requested shape", () => {
  const matrix = randomNormalMatrix(3, 5, 0.02, createSeededRandom(1));

  assert.equal(matrix.length, 3);
  assert.ok(matrix.every((row) => row.length === 5));
});

test("residualProjectionStandardDeviation() divides by sqrt(2 * numberTransformerBlocks)", () => {
  assert.equal(residualProjectionStandardDeviation(0.02, 12), 0.02 / Math.sqrt(24));
});

test("every layer of the model is initialized as in GPT-2", () => {
  // embeddingSize 64, 4 heads, 6 blocks, with query/key/value biases switched on so they can be checked.
  const hyperparameters = new Hyperparameters(64, 32, 0.1, Hyperparameters.TRAINING_MODE, 4, 6, 2, true);
  hyperparameters.seed = 7;
  const model = new GPTModel(hyperparameters);
  const residual = 0.02 / Math.sqrt(2 * 6);

  // Embeddings and the output layer.
  const embedding = model.embedding as unknown as { textEmbeddingMatrix: number[][]; positionEmbeddingMatrix: number[][] };
  assertNormal(embedding.textEmbeddingMatrix, 0.02, "token embedding");
  assertNormal(embedding.positionEmbeddingMatrix, 0.01, "position embedding");
  assertNormal(model.linearOutputLayer.weights, 0.02, "output layer");

  // Every transformer block.
  model.transformers.forEach((transformer, block) => {
    const heads = transformer.multiHeadAttention.attentionHeads;
    for (const name of ["queryWeights", "keyWeights", "valueWeights"] as const) {
      assertNormal(heads.flatMap((head) => head[name]), 0.02, `block ${block} ${name}`);
    }
    for (const name of ["queryBiases", "keyBiases", "valueBiases"] as const) {
      assertAllZero(heads.flatMap((head) => head[name]), `block ${block} ${name}`);
    }

    assertNormal(transformer.multiHeadAttention.outputProjectionWeights, residual, `block ${block} attention output projection`);
    assertAllZero(transformer.multiHeadAttention.outputProjectionBiases, `block ${block} attention output projection biases`);

    assertNormal(transformer.feedForward.firstLayerWeights, 0.02, `block ${block} feed forward first layer`);
    assertNormal(transformer.feedForward.secondLayerWeights, residual, `block ${block} feed forward second layer`);
    assertAllZero(transformer.feedForward.firstLayerBiases, `block ${block} feed forward first layer biases`);
    assertAllZero(transformer.feedForward.secondLayerBiases, `block ${block} feed forward second layer biases`);
  });
});

test("linearLayerStandardDeviation controls every linear layer in the transformer blocks", () => {
  const hyperparameters = new Hyperparameters(64, 8, 0.1, Hyperparameters.TRAINING_MODE, 4, 2);
  hyperparameters.linearLayerStandardDeviation = 0.05;
  hyperparameters.seed = 3;
  const transformer = new GPTModel(hyperparameters).transformers[0];

  assertNormal(transformer.multiHeadAttention.attentionHeads.flatMap((head) => head.queryWeights), 0.05, "query weights");
  assertNormal(transformer.feedForward.firstLayerWeights, 0.05, "feed forward first layer");
  assertNormal(transformer.feedForward.secondLayerWeights, 0.05 / Math.sqrt(4), "feed forward second layer");
});
