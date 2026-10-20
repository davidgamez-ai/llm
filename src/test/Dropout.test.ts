import { test } from "node:test";
import assert from "node:assert/strict";
import Hyperparameters from "../Hyperparameters.js";
import Dropout from "../Dropout.js";

// A 3 x 4 matrix of distinct, non-zero values, so dropped values are easy to spot.
const VALUES: number[][] = [
  [1, 2, 3, 4],
  [5, 6, 7, 8],
  [9, 10, 11, 12],
];

/**
 * Creates a Dropout in the given mode with a seeded random stream.
 *
 * @param mode - TRAINING_MODE, VALIDATION_MODE or GENERATION_MODE.
 * @param dropoutRate - The probability of dropping each value.
 * @returns The Dropout and the Hyperparameters it reads its mode from.
 */
const setUp = (mode: number, dropoutRate: number = 0.5) => {
  const hyperparameters = new Hyperparameters();
  hyperparameters.mode = mode;
  hyperparameters.dropoutRate = dropoutRate;
  hyperparameters.seed = 7;
  return { hyperparameters, dropout: new Dropout(hyperparameters) };
};

test("in validation and generation modes, update() returns the values unchanged and stores no mask", () => {
  for (const mode of [Hyperparameters.VALIDATION_MODE, Hyperparameters.GENERATION_MODE]) {
    const { dropout } = setUp(mode);

    assert.equal(dropout.update(VALUES), VALUES);
    assert.equal(dropout.mask, null);
  }
});

test("in training mode every value is either dropped or scaled by 1 / (1 - dropoutRate)", () => {
  const { dropout } = setUp(Hyperparameters.TRAINING_MODE, 0.5);

  const output = dropout.update(VALUES);

  let dropped = 0;
  for (let i = 0; i < VALUES.length; i++) {
    for (let j = 0; j < VALUES[i].length; j++) {
      // With a dropout rate of 0.5, kept values are doubled.
      assert.ok(output[i][j] === 0 || output[i][j] === 2 * VALUES[i][j], `value (${i}, ${j}) is ${output[i][j]}`);
      if (output[i][j] === 0) dropped++;
    }
  }
  // With 12 values and seed 7, some but not all are dropped.
  assert.ok(dropped > 0 && dropped < 12, `${dropped} values dropped`);
});

test("the stored mask matches the output", () => {
  const { dropout } = setUp(Hyperparameters.TRAINING_MODE, 0.5);

  const output = dropout.update(VALUES);
  const mask = dropout.mask!;

  assert.deepEqual(output, VALUES.map((row, i) => row.map((value, j) => value * mask[i][j])));
  assert.ok(mask.every((row) => row.every((entry) => entry === 0 || entry === 2)));
});

test("backpropagate() passes no error to dropped values and scales the rest by the same mask", () => {
  const { dropout } = setUp(Hyperparameters.TRAINING_MODE, 0.5);
  const output = dropout.update(VALUES);
  const errors = VALUES.map((row) => row.map(() => 1));

  const inputErrors = dropout.backpropagate(errors);

  for (let i = 0; i < VALUES.length; i++) {
    for (let j = 0; j < VALUES[i].length; j++) {
      // A dropped value (output 0) gets error 0; a kept value gets 1 x 2.
      assert.equal(inputErrors[i][j], output[i][j] === 0 ? 0 : 2);
    }
  }
});

test("backpropagate() returns the errors unchanged when the last update() was not in training mode", () => {
  const { hyperparameters, dropout } = setUp(Hyperparameters.TRAINING_MODE);
  dropout.update(VALUES);

  // A later update() in validation mode clears the mask.
  hyperparameters.mode = Hyperparameters.VALIDATION_MODE;
  dropout.update(VALUES);
  const errors = VALUES.map((row) => row.map(() => 1));

  assert.equal(dropout.backpropagate(errors), errors);
});

test("backpropagate() rejects errors of a different shape from the mask", () => {
  const { dropout } = setUp(Hyperparameters.TRAINING_MODE);
  dropout.update(VALUES);

  assert.throws(() => dropout.backpropagate([[1, 2, 3, 4]]), RangeError);
  assert.throws(() => dropout.backpropagate(VALUES.map((row) => row.slice(1))), RangeError);
});

test("a dropout rate of 0 keeps every value unchanged in training mode", () => {
  const { dropout } = setUp(Hyperparameters.TRAINING_MODE, 0);

  assert.deepEqual(dropout.update(VALUES), VALUES);
});
