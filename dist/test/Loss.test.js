import { test } from "node:test";
import assert from "node:assert/strict";
import { softmax, crossEntropyLoss, crossEntropyErrors } from "../Loss.js";
// A small row of logits with a mix of signs and sizes.
const LOGITS = [2.0, -1.0, 0.5, 3.0, 0.0];
test("softmax() gives positive values that sum to 1, in the same order as the logits", () => {
    const probabilities = softmax(LOGITS);
    assert.ok(probabilities.every((p) => p > 0));
    assert.ok(Math.abs(probabilities.reduce((sum, p) => sum + p, 0) - 1) < 1e-12);
    // The largest logit (index 3) has the largest probability.
    assert.equal(probabilities.indexOf(Math.max(...probabilities)), 3);
});
test("softmax() handles very large logits without overflow", () => {
    const probabilities = softmax([1000, 1000, 999]);
    assert.ok(probabilities.every((p) => Number.isFinite(p)));
    assert.ok(Math.abs(probabilities[0] - probabilities[1]) < 1e-12);
});
test("softmax() gives exactly 0 to -Infinity, as used by the causal mask", () => {
    assert.deepEqual(softmax([0, -Infinity]), [1, 0]);
});
test("crossEntropyLoss() of equal logits is ln(number of logits)", () => {
    const vocabularySize = 50257;
    const logits = new Array(vocabularySize).fill(0.3);
    assert.ok(Math.abs(crossEntropyLoss(logits, 123) - Math.log(vocabularySize)) < 1e-9);
});
test("crossEntropyLoss() equals -log(softmax(logits)[target])", () => {
    for (let target = 0; target < LOGITS.length; target++) {
        assert.ok(Math.abs(crossEntropyLoss(LOGITS, target) + Math.log(softmax(LOGITS)[target])) < 1e-12);
    }
});
test("crossEntropyLoss() stays finite when the target's probability is tiny", () => {
    // exp(-2000) underflows to 0, so log(softmax) would be -Infinity; the log-softmax is not.
    assert.ok(Math.abs(crossEntropyLoss([0, -2000], 1) - 2000) < 1e-9);
});
test("crossEntropyErrors() is the probability minus the one-hot target", () => {
    const probabilities = softmax(LOGITS);
    const errors = crossEntropyErrors(LOGITS, 1);
    errors.forEach((error, i) => {
        assert.ok(Math.abs(error - (probabilities[i] - (i === 1 ? 1 : 0))) < 1e-12);
    });
});
test("crossEntropyErrors() sum to 0, and only the target's error is negative", () => {
    const errors = crossEntropyErrors(LOGITS, 2);
    assert.ok(Math.abs(errors.reduce((sum, e) => sum + e, 0)) < 1e-12);
    errors.forEach((error, i) => assert.ok(i === 2 ? error < 0 : error > 0, `error ${i} is ${error}`));
});
test("crossEntropyErrors() multiplies every error by scale", () => {
    const unscaled = crossEntropyErrors(LOGITS, 0);
    const scaled = crossEntropyErrors(LOGITS, 0, 0.25);
    scaled.forEach((error, i) => assert.ok(Math.abs(error - 0.25 * unscaled[i]) < 1e-12));
});
test("crossEntropyErrors() matches the numerical gradient of crossEntropyLoss()", () => {
    /*
     * Finite-difference check: nudge each logit up and down by h and measure
     * how much the loss changes. (L(x + h) - L(x - h)) / 2h approximates the
     * derivative, with an error proportional to h squared.
     */
    const h = 1e-5;
    const target = 3;
    const errors = crossEntropyErrors(LOGITS, target);
    for (let i = 0; i < LOGITS.length; i++) {
        const plus = [...LOGITS];
        const minus = [...LOGITS];
        plus[i] += h;
        minus[i] -= h;
        const numerical = (crossEntropyLoss(plus, target) - crossEntropyLoss(minus, target)) / (2 * h);
        assert.ok(Math.abs(numerical - errors[i]) < 1e-8, `logit ${i}: analytic ${errors[i]}, numerical ${numerical}`);
    }
});
test("crossEntropyLoss() and crossEntropyErrors() reject a target with no logit", () => {
    for (const target of [-1, LOGITS.length, 1.5, NaN]) {
        assert.throws(() => crossEntropyLoss(LOGITS, target), RangeError);
        assert.throws(() => crossEntropyErrors(LOGITS, target), RangeError);
    }
});
