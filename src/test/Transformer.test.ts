import { test } from "node:test";
import assert from "node:assert/strict";
import Hyperparameters from "../Hyperparameters.js";
import Transformer from "../Transformer.js";
import createSeededRandom from "../SeededRandom.js";
import { checkGradients, randomMatrix } from "./GradientCheck.js";

/*
 * Builds a Transformer block with 2 heads over embeddings of 8 values. The
 * weights use a larger standard deviation (0.5) than GPT-2's 0.02, so every
 * stage has a real effect, and the layer normalizations' scale and shift are
 * set to random values.
 */
const buildBlock = (dropoutRate: number = 0) => {
  const hyperparameters = new Hyperparameters(8, 4, dropoutRate, Hyperparameters.TRAINING_MODE, 2, 1);
  hyperparameters.linearLayerStandardDeviation = 0.5;
  hyperparameters.seed = 61;
  const block = new Transformer(hyperparameters);
  const random = createSeededRandom(62);
  for (const layerNormalization of [block.attentionLayerNormalization, block.feedForwardLayerNormalization]) {
    layerNormalization.scale.forEach((_, i) => (layerNormalization.scale[i] = 0.5 + random()));
    layerNormalization.shift.forEach((_, i) => (layerNormalization.shift[i] = random() - 0.5));
  }
  return { hyperparameters, block };
};

/**
 * Runs the gradient check on a block, for one parameter array from each of
 * its trainable stages, and its inputs.
 *
 * @param block - The block to check.
 * @param beforeForward - Called before every forward pass, e.g. to reset the random seed so dropout draws the same masks each time.
 */
const check = (block: Transformer, beforeForward: () => void = () => {}): void => {
  const attention = block.multiHeadAttention;
  const head = attention.attentionHeads[0];
  const inputs = randomMatrix(4, 8, 63, 1.5);
  let inputErrors: number[][] = [];

  checkGradients({
    forward: () => {
      beforeForward();
      return block.update(inputs);
    },
    backward: (outputErrors) => {
      inputErrors = block.backpropagate(outputErrors);
      block.calculateGradients(false);
    },
    parameters: [
      { name: "attention layer normalization scale", values: block.attentionLayerNormalization.scale, gradients: () => block.attentionLayerNormalization.scaleGradients },
      { name: "head 0 keyWeights", values: head.keyWeights.data, gradients: () => head.keyWeightGradients.data },
      { name: "attention outputProjectionWeights", values: attention.outputProjectionWeights.data, gradients: () => attention.outputProjectionWeightGradients.data },
      { name: "feed forward layer normalization shift", values: block.feedForwardLayerNormalization.shift, gradients: () => block.feedForwardLayerNormalization.shiftGradients },
      { name: "feed forward firstLayerWeights", values: block.feedForward.firstLayerWeights.data, gradients: () => block.feedForward.firstLayerWeightGradients.data },
      { name: "feed forward secondLayerBiases", values: block.feedForward.secondLayerBiases, gradients: () => block.feedForward.secondLayerBiasGradients },
    ],
    inputs,
    inputErrors: () => inputErrors,
  });
};

test("backpropagation gives the same gradients in every stage, and input errors, as the numerical gradients", () => {
  check(buildBlock().block);
});

test("backpropagation through both dropouts matches the numerical gradients", () => {
  const { hyperparameters, block } = buildBlock(0.3);

  // Resetting the seed before every forward pass makes every dropout draw the same mask each time.
  check(block, () => { hyperparameters.seed = 64; });
});

test("the shortcut connections pass the output errors straight through to the input", () => {
  const { block } = buildBlock();

  /*
   * With the last layer of each branch (the attention output projection and
   * the feed forward second layer) set to 0, the branches output nothing
   * and pass no errors back, so the only route left is the shortcuts, which
   * copy the errors unchanged.
   */
  block.multiHeadAttention.outputProjectionWeights.data.fill(0);
  block.feedForward.secondLayerWeights.data.fill(0);

  block.update(randomMatrix(4, 8, 65));
  const outputErrors = randomMatrix(4, 8, 66);

  assert.deepEqual(block.backpropagate(outputErrors), outputErrors);
});
