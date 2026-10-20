import { test } from "node:test";
import assert from "node:assert/strict";
import Hyperparameters from "../Hyperparameters.js";
import LayerNormalization from "../LayerNormalization.js";
import createSeededRandom from "../SeededRandom.js";
import { checkGradients, randomMatrix } from "./GradientCheck.js";

/*
 * Builds a LayerNormalization over 8 values. scale and shift are set to
 * random values, rather than their initial 1s and 0s, so the tests cover
 * the general case.
 */
const buildLayer = (): LayerNormalization => {
  const layer = new LayerNormalization(new Hyperparameters(8, 4, 0, Hyperparameters.TRAINING_MODE, 2, 1));
  const random = createSeededRandom(21);
  layer.scale.forEach((_, i) => (layer.scale[i] = 0.5 + random()));
  layer.shift.forEach((_, i) => (layer.shift[i] = random() - 0.5));
  return layer;
};

test("with scale 1 and shift 0, update() gives each token a mean of 0 and a variance of 1", () => {
  const layer = new LayerNormalization(new Hyperparameters(8, 4, 0, Hyperparameters.TRAINING_MODE, 2, 1));

  for (const row of layer.update(randomMatrix(3, 8, 1, 5))) {
    const mean = row.reduce((sum, value) => sum + value, 0) / row.length;
    const variance = row.reduce((sum, value) => sum + (value - mean) ** 2, 0) / row.length;
    assert.ok(Math.abs(mean) < 1e-9);
    assert.ok(Math.abs(variance - 1) < 1e-3);
  }
});

test("backpropagation gives the same scale and shift gradients and input errors as the numerical gradients", () => {
  const layer = buildLayer();
  // Inputs with a non-zero mean and a spread bigger than 1, as real activations have.
  const inputs = randomMatrix(4, 8, 2, 3).map((row) => row.map((value) => value + 1));
  let inputErrors: number[][] = [];

  checkGradients({
    forward: () => layer.update(inputs),
    backward: (outputErrors) => {
      inputErrors = layer.backpropagate(outputErrors);
      layer.calculateGradients(false);
    },
    parameters: [
      { name: "scale", values: layer.scale, gradients: () => layer.scaleGradients },
      { name: "shift", values: layer.shift, gradients: () => layer.shiftGradients },
    ],
    inputs,
    inputErrors: () => inputErrors,
  });
});

test("the input errors of each token sum to 0, since shifting every input equally changes nothing", () => {
  const layer = buildLayer();
  layer.update(randomMatrix(4, 8, 3, 2));

  for (const row of layer.backpropagate(randomMatrix(4, 8, 4))) {
    assert.ok(Math.abs(row.reduce((sum, error) => sum + error, 0)) < 1e-9);
  }
});
