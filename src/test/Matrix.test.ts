import { test } from "node:test";
import assert from "node:assert/strict";
import Matrix, { vectorTimesMatrix, matrixTimesVector } from "../Matrix.js";

// The 2 x 3 matrix [[1, 2, 3], [4, 5, 6]], used by several tests.
const twoByThree = (): Matrix => Matrix.fromArrays([[1, 2, 3], [4, 5, 6]]);

test("a new Matrix has the given shape and is filled with zeros", () => {
  const matrix = new Matrix(2, 3);

  assert.equal(matrix.rows, 2);
  assert.equal(matrix.columns, 3);
  assert.equal(matrix.size, 6);
  assert.ok(matrix.data.every((value) => value === 0));
});

test("values are stored row by row", () => {
  assert.deepEqual(Array.from(twoByThree().data), [1, 2, 3, 4, 5, 6]);
});

test("get() and set() read and change the value at (row, column)", () => {
  const matrix = twoByThree();

  assert.equal(matrix.get(1, 2), 6);
  matrix.set(1, 2, 60);
  assert.equal(matrix.get(1, 2), 60);
  assert.equal(matrix.data[5], 60);
});

test("getRow() returns a view onto the row, not a copy", () => {
  const matrix = twoByThree();
  const row = matrix.getRow(1);

  assert.deepEqual(Array.from(row), [4, 5, 6]);
  row[0] = 40;
  assert.equal(matrix.get(1, 0), 40);
});

test("toArrays() copies the matrix back into nested arrays", () => {
  assert.deepEqual(twoByThree().toArrays(), [[1, 2, 3], [4, 5, 6]]);
});

test("the constructor and fromArrays() reject data of the wrong size", () => {
  assert.throws(() => new Matrix(2, 3, new Float32Array(5)), RangeError);
  assert.throws(() => Matrix.fromArrays([[1, 2], [3]]), RangeError);
});

test("values are stored as 32-bit floats", () => {
  const matrix = new Matrix(1, 1);

  matrix.set(0, 0, 0.1);

  assert.equal(matrix.get(0, 0), Math.fround(0.1));
});

test("vectorTimesMatrix() multiplies a vector by a matrix", () => {
  // [1, 2] x [[1, 2, 3], [4, 5, 6]] = [1*1 + 2*4, 1*2 + 2*5, 1*3 + 2*6] = [9, 12, 15]
  assert.deepEqual(vectorTimesMatrix([1, 2], twoByThree()), [9, 12, 15]);
});

test("vectorTimesMatrix() adds the biases, and treats empty biases as none", () => {
  assert.deepEqual(vectorTimesMatrix([1, 2], twoByThree(), new Float32Array([10, 20, 30])), [19, 32, 45]);
  assert.deepEqual(vectorTimesMatrix([1, 2], twoByThree(), new Float32Array(0)), [9, 12, 15]);
});

test("vectorTimesMatrix() rejects a vector or biases of the wrong length", () => {
  assert.throws(() => vectorTimesMatrix([1, 2, 3], twoByThree()), RangeError);
  assert.throws(() => vectorTimesMatrix([1, 2], twoByThree(), [1, 2]), RangeError);
});

test("matrixTimesVector() takes the dot product of each row with the vector", () => {
  // [[1, 2, 3], [4, 5, 6]] x [1, 0, 2] = [1 + 6, 4 + 12] = [7, 16]
  assert.deepEqual(matrixTimesVector(twoByThree(), [1, 0, 2]), [7, 16]);
});

test("matrixTimesVector() rejects a vector of the wrong length", () => {
  assert.throws(() => matrixTimesVector(twoByThree(), [1, 2]), RangeError);
});
