import { test } from "node:test";
import assert from "node:assert/strict";
import Matrix, { vectorTimesMatrix, matrixTimesVector, addOuterProduct, addToVector } from "../Matrix.js";
// The 2 x 3 matrix [[1, 2, 3], [4, 5, 6]], used by several tests.
const twoByThree = () => Matrix.fromArrays([[1, 2, 3], [4, 5, 6]]);
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
test("matrixTimesVector(W, e) equals vectorTimesMatrix(e, W transposed), as used in backpropagation", () => {
    const matrix = twoByThree();
    // Build the transpose by hand: 3 x 2, with (row, column) = (column, row) of the original.
    const transpose = new Matrix(3, 2);
    for (let row = 0; row < 2; row++) {
        for (let column = 0; column < 3; column++) {
            transpose.set(column, row, matrix.get(row, column));
        }
    }
    const errors = [0.5, -2, 3];
    assert.deepEqual(matrixTimesVector(matrix, errors), vectorTimesMatrix(errors, transpose));
});
test("addOuterProduct() adds input[row] * errors[column] to every gradient", () => {
    const gradients = new Matrix(2, 3);
    // [1, 2] outer [3, 4, 5] = [[3, 4, 5], [6, 8, 10]]
    addOuterProduct(gradients, [1, 2], [3, 4, 5]);
    assert.deepEqual(gradients.toArrays(), [[3, 4, 5], [6, 8, 10]]);
    // A second token's contribution is added to the first, not replacing it.
    addOuterProduct(gradients, [1, 0], [1, 1, 1]);
    assert.deepEqual(gradients.toArrays(), [[4, 5, 6], [6, 8, 10]]);
});
test("addOuterProduct() rejects an input or errors of the wrong length", () => {
    assert.throws(() => addOuterProduct(new Matrix(2, 3), [1, 2, 3], [1, 2, 3]), RangeError);
    assert.throws(() => addOuterProduct(new Matrix(2, 3), [1, 2], [1, 2]), RangeError);
});
test("addToVector() adds the values element by element, in place", () => {
    const target = new Float32Array([1, 2, 3]);
    addToVector(target, [10, 20, 30]);
    addToVector(target, [1, 1, 1]);
    assert.deepEqual(Array.from(target), [12, 23, 34]);
});
test("addToVector() rejects values of the wrong length", () => {
    assert.throws(() => addToVector(new Float32Array(3), [1, 2]), RangeError);
});
