import { test } from "node:test";
import assert from "node:assert/strict";
import Matrix, { vectorTimesMatrix, matrixTimesVector, addOuterProduct } from "../Matrix.js";
import { randomNormalMatrix } from "../WeightInitialization.js";
import createSeededRandom from "../SeededRandom.js";
import { checkGradients } from "./GradientCheck.js";

/*
 * Tests for the gradient-checking helper itself, using small fake
 * components whose correct gradients are known. A helper that passed
 * everything would be useless, so the tests also check that it fails when
 * a gradient is wrong.
 */

/**
 * A fake component: a linear layer with no bias, output = input x weights,
 * optionally followed by tanh. It stores its inputs and outputs in update()
 * and calculates its gradients in backpropagate(), as the real components
 * will.
 */
class FakeLayer {
  weights: Matrix;
  weightGradients: Matrix;
  inputErrors: number[][] = [];
  private inputs: number[][] = [];
  private outputs: number[][] = [];

  /**
   * @param useTanh - True to apply tanh after the linear layer, to test a non-linear component.
   * @param weightGradientFactor - Multiplies the weight gradients. 1 is correct; anything else is a deliberate bug.
   * @param inputErrorFactor - Multiplies the input errors. 1 is correct; anything else is a deliberate bug.
   */
  constructor(
    private readonly useTanh: boolean,
    private readonly weightGradientFactor: number = 1,
    private readonly inputErrorFactor: number = 1
  ) {
    this.weights = randomNormalMatrix(3, 2, 0.5, createSeededRandom(5));
    this.weightGradients = new Matrix(3, 2);
  }

  update(inputs: number[][]): number[][] {
    this.inputs = inputs;
    this.outputs = inputs.map((input) => {
      const linear = vectorTimesMatrix(input, this.weights);
      return this.useTanh ? linear.map(Math.tanh) : linear;
    });
    return this.outputs;
  }

  backpropagate(outputErrors: number[][]): void {
    // Through tanh: d tanh(z)/dz = 1 - tanh(z)^2, using the stored outputs.
    const linearErrors = this.useTanh
      ? outputErrors.map((row, t) => row.map((error, j) => error * (1 - this.outputs[t][j] ** 2)))
      : outputErrors;

    this.weightGradients = new Matrix(3, 2);
    for (let t = 0; t < this.inputs.length; t++) {
      addOuterProduct(this.weightGradients, this.inputs[t], linearErrors[t]);
    }
    this.weightGradients.data.forEach((_, i) => (this.weightGradients.data[i] *= this.weightGradientFactor));

    this.inputErrors = linearErrors.map((errors) =>
      matrixTimesVector(this.weights, errors).map((error) => error * this.inputErrorFactor)
    );
  }
}

/**
 * Runs checkGradients on a FakeLayer with four tokens of three inputs.
 *
 * @param layer - The layer to check.
 * @returns The number of values checked.
 */
const check = (layer: FakeLayer): number => {
  const inputs: number[][] = [
    [0.5, -1.0, 0.25],
    [1.5, 0.3, -0.7],
    [-0.2, 0.8, 1.1],
    [0.0, -0.4, 0.6],
  ];
  return checkGradients({
    forward: () => layer.update(inputs),
    backward: (outputErrors) => layer.backpropagate(outputErrors),
    parameters: [{ name: "weights", values: layer.weights.data, gradients: () => layer.weightGradients.data }],
    inputs,
    inputErrors: () => layer.inputErrors,
  });
};

test("checkGradients() passes a linear layer with correct gradients, checking every value", () => {
  // 6 weights and 12 inputs, all below the default sample size of 20.
  assert.equal(check(new FakeLayer(false)), 18);
});

test("checkGradients() passes a non-linear layer with correct gradients", () => {
  check(new FakeLayer(true));
});

test("checkGradients() fails when a weight gradient is wrong", () => {
  assert.throws(() => check(new FakeLayer(true, 2)), /gradients disagree[\s\S]*weights\[/);
});

test("checkGradients() fails when an input error is wrong", () => {
  assert.throws(() => check(new FakeLayer(true, 1, 1.1)), /gradients disagree[\s\S]*inputs\[/);
});

test("checkGradients() leaves the parameters and inputs as it found them", () => {
  const layer = new FakeLayer(true);
  const weightsBefore = Array.from(layer.weights.data);
  const inputs = [[0.5, -1.0, 0.25], [1.5, 0.3, -0.7]];
  const inputsBefore = inputs.map((row) => [...row]);

  checkGradients({
    forward: () => layer.update(inputs),
    backward: (outputErrors) => layer.backpropagate(outputErrors),
    parameters: [{ name: "weights", values: layer.weights.data, gradients: () => layer.weightGradients.data }],
    inputs,
    inputErrors: () => layer.inputErrors,
  });

  assert.deepEqual(Array.from(layer.weights.data), weightsBefore);
  assert.deepEqual(inputs, inputsBefore);
});

test("checkGradients() samples large arrays rather than checking every value", () => {
  const values = new Float32Array(1000).fill(0.5);

  // output = 2 * sum of values, so every gradient is 2 times the single output error.
  let gradient = 0;
  const checked = checkGradients({
    forward: () => [[2 * values.reduce((sum, v) => sum + v, 0)]],
    backward: (outputErrors) => { gradient = 2 * outputErrors[0][0]; },
    parameters: [{ name: "values", values, gradients: () => new Float32Array(1000).fill(gradient) }],
    samplesPerArray: 5,
  });

  assert.equal(checked, 5);
});
