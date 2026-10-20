import { test } from "node:test";
import assert from "node:assert/strict";
import Hyperparameters from "../Hyperparameters.js";
import Matrix from "../Matrix.js";
import AdamW from "../AdamW.js";
/** Creates an optimizer with the default hyperparameters (learning rate 0.0004, weight decay 0.1). */
const setUp = () => {
    const hyperparameters = new Hyperparameters();
    return { hyperparameters, optimizer: new AdamW(hyperparameters) };
};
/**
 * Asserts that two numbers are within tolerance of each other. Parameters
 * are stored as 32-bit floats, so they only match a 64-bit calculation to
 * about 7 significant digits.
 */
const assertClose = (actual, expected, tolerance = 1e-7, label = "") => {
    assert.ok(Math.abs(actual - expected) <= tolerance, `${label} expected ${expected}, got ${actual}`);
};
test("updating before step() throws", () => {
    const { optimizer } = setUp();
    assert.throws(() => optimizer.updateVector(new Float32Array(2), new Float32Array(2)), /Call step\(\)/);
});
test("step() counts the steps", () => {
    const { optimizer } = setUp();
    optimizer.step();
    optimizer.step();
    assert.equal(optimizer.stepCount, 2);
});
test("on the first step every parameter moves by learningRate against its gradient, whatever the gradient's size", () => {
    const { optimizer } = setUp();
    const parameters = new Float32Array([1, 1, 1, 1]);
    // Gradients of very different sizes and both signs.
    const gradients = new Float32Array([0.001, 5, -0.5, -200]);
    optimizer.step();
    optimizer.updateVector(parameters, gradients);
    // With bias correction, mHat / sqrt(vHat) is exactly +1 or -1 on the first step.
    const expected = [1 - 0.0004, 1 - 0.0004, 1 + 0.0004, 1 + 0.0004];
    parameters.forEach((value, i) => assertClose(value, expected[i], 1e-7, `parameter ${i}`));
});
test("a zero gradient leaves a parameter unchanged without weight decay, and shrinks it with weight decay", () => {
    const { optimizer } = setUp();
    const withoutDecay = new Float32Array([2, -3]);
    const withDecay = new Float32Array([2, -3]);
    optimizer.step();
    optimizer.updateVector(withoutDecay, new Float32Array(2), false);
    optimizer.updateVector(withDecay, new Float32Array(2), true);
    assert.deepEqual(Array.from(withoutDecay), [2, -3]);
    // Each value is multiplied by 1 - learningRate * weightDecay = 1 - 0.0004 * 0.1.
    assertClose(withDecay[0], 2 * (1 - 0.00004));
    assertClose(withDecay[1], -3 * (1 - 0.00004));
});
test("three steps match a hand calculation of the AdamW formula", () => {
    const { optimizer } = setUp();
    const parameters = new Float32Array([0.5]);
    const gradientsByStep = [0.2, -0.1, 0.4];
    // The formula, worked through in 64-bit with the default hyperparameters.
    let p = 0.5, m = 0, v = 0;
    gradientsByStep.forEach((g, index) => {
        const t = index + 1;
        m = 0.9 * m + 0.1 * g;
        v = 0.999 * v + 0.001 * g * g;
        const mHat = m / (1 - 0.9 ** t);
        const vHat = v / (1 - 0.999 ** t);
        p -= 0.0004 * 0.1 * p;
        p -= (0.0004 * mHat) / (Math.sqrt(vHat) + 1e-8);
        optimizer.step();
        optimizer.updateVector(parameters, new Float32Array([g]), true);
        assertClose(parameters[0], p, 1e-7, `step ${t}`);
    });
});
test("momentum: a gradient that reverses sign does not immediately reverse the parameter's direction", () => {
    const { optimizer } = setUp();
    const parameters = new Float32Array([0]);
    // Ten steps pushing the parameter down...
    for (let t = 0; t < 10; t++) {
        optimizer.step();
        optimizer.updateVector(parameters, new Float32Array([1]));
    }
    const before = parameters[0];
    // ...then one step with the gradient reversed. The running average is still positive, so the parameter keeps falling.
    optimizer.step();
    optimizer.updateVector(parameters, new Float32Array([-1]));
    assert.ok(parameters[0] < before, `${parameters[0]} should be below ${before}`);
});
test("each parameter array has its own running averages", () => {
    const { optimizer } = setUp();
    const up = new Float32Array([0]);
    const down = new Float32Array([0]);
    /*
     * With a constant gradient, mHat / sqrt(vHat) is exactly 1 at every step,
     * so each array moves by learningRate per step. If the two arrays shared
     * their running averages, their opposite gradients would mix and they
     * would barely move.
     */
    for (let t = 0; t < 5; t++) {
        optimizer.step();
        optimizer.updateVector(up, new Float32Array([-1]));
        optimizer.updateVector(down, new Float32Array([1]));
    }
    assertClose(up[0], 5 * 0.0004, 1e-7, "up");
    assertClose(down[0], -5 * 0.0004, 1e-7, "down");
});
test("updateMatrix() updates every value of the matrix, with weight decay by default", () => {
    const { optimizer } = setUp();
    const weights = Matrix.fromArrays([[1, 2], [3, 4]]);
    optimizer.step();
    optimizer.updateMatrix(weights, new Matrix(2, 2));
    // Zero gradients, so only weight decay changes the values.
    [1, 2, 3, 4].forEach((value, i) => assertClose(weights.data[i], value * (1 - 0.00004), 1e-6));
});
test("the learning rate is read on every update, so changing it takes effect immediately", () => {
    const { hyperparameters, optimizer } = setUp();
    const parameters = new Float32Array([1]);
    hyperparameters.learningRate = 0.01;
    optimizer.step();
    optimizer.updateVector(parameters, new Float32Array([1]));
    assertClose(parameters[0], 0.99);
});
test("gradients of the wrong size are rejected", () => {
    const { optimizer } = setUp();
    optimizer.step();
    assert.throws(() => optimizer.updateVector(new Float32Array(3), new Float32Array(2)), RangeError);
    assert.throws(() => optimizer.updateMatrix(new Matrix(2, 3), new Matrix(3, 2)), RangeError);
});
