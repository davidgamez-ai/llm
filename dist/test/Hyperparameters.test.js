import { test } from "node:test";
import assert from "node:assert/strict";
import Hyperparameters from "../Hyperparameters.js";
// Values that fail the positive whole number check: zero, negative, fractional, NaN and Infinity.
const NON_POSITIVE_INTEGERS = [0, -1, 1.5, NaN, Infinity];
test("unvalidated setters store the new value", () => {
    const hyperparameters = new Hyperparameters();
    hyperparameters.qkvBias = true;
    hyperparameters.feedForwardBias = false;
    assert.equal(hyperparameters.qkvBias, true);
    assert.equal(hyperparameters.feedForwardBias, false);
});
test("contextLength setter stores a positive whole number", () => {
    const hyperparameters = new Hyperparameters();
    hyperparameters.contextLength = 256;
    assert.equal(hyperparameters.contextLength, 256);
});
test("contextLength setter rejects invalid values and leaves state unchanged", () => {
    const hyperparameters = new Hyperparameters();
    for (const value of NON_POSITIVE_INTEGERS) {
        assert.throws(() => { hyperparameters.contextLength = value; }, RangeError, `accepted ${value}`);
    }
    assert.equal(hyperparameters.contextLength, 12);
});
test("trainingRatio defaults to 0.9 and numberEpochs to 1", () => {
    const hyperparameters = new Hyperparameters();
    assert.equal(hyperparameters.trainingRatio, 0.9);
    assert.equal(hyperparameters.numberEpochs, 1);
});
test("trainingRatio setter stores values between 0 and 1", () => {
    const hyperparameters = new Hyperparameters();
    hyperparameters.trainingRatio = 0.75;
    assert.equal(hyperparameters.trainingRatio, 0.75);
});
test("trainingRatio setter rejects 0, 1, values outside them and NaN, leaving state unchanged", () => {
    const hyperparameters = new Hyperparameters();
    for (const value of [0, 1, -0.1, 1.1, NaN]) {
        assert.throws(() => { hyperparameters.trainingRatio = value; }, RangeError, `accepted ${value}`);
    }
    assert.equal(hyperparameters.trainingRatio, 0.9);
});
test("numberEpochs setter stores a positive whole number", () => {
    const hyperparameters = new Hyperparameters();
    hyperparameters.numberEpochs = 10;
    assert.equal(hyperparameters.numberEpochs, 10);
});
test("numberEpochs setter rejects invalid values and leaves state unchanged", () => {
    const hyperparameters = new Hyperparameters();
    for (const value of NON_POSITIVE_INTEGERS) {
        assert.throws(() => { hyperparameters.numberEpochs = value; }, RangeError, `accepted ${value}`);
    }
    assert.equal(hyperparameters.numberEpochs, 1);
});
test("constructor rejects an invalid contextLength", () => {
    for (const value of NON_POSITIVE_INTEGERS) {
        assert.throws(() => new Hyperparameters(768, value), RangeError, `accepted ${value}`);
    }
});
test("mode defaults to TRAINING_MODE", () => {
    assert.equal(new Hyperparameters().mode, Hyperparameters.TRAINING_MODE);
});
test("the three modes have distinct values", () => {
    const modes = [Hyperparameters.TRAINING_MODE, Hyperparameters.VALIDATION_MODE, Hyperparameters.GENERATION_MODE];
    assert.equal(new Set(modes).size, 3);
});
test("mode setter stores each of the three modes", () => {
    const hyperparameters = new Hyperparameters();
    for (const mode of [Hyperparameters.VALIDATION_MODE, Hyperparameters.GENERATION_MODE, Hyperparameters.TRAINING_MODE]) {
        hyperparameters.mode = mode;
        assert.equal(hyperparameters.mode, mode);
    }
});
test("mode setter rejects values that are not one of the three modes, leaving state unchanged", () => {
    const hyperparameters = new Hyperparameters();
    hyperparameters.mode = Hyperparameters.VALIDATION_MODE;
    for (const value of [-1, 3, 0.5, NaN]) {
        assert.throws(() => { hyperparameters.mode = value; }, RangeError, `accepted ${value}`);
    }
    assert.equal(hyperparameters.mode, Hyperparameters.VALIDATION_MODE);
});
test("constructor rejects an invalid mode", () => {
    assert.throws(() => new Hyperparameters(768, 1024, 0.1, 3), RangeError);
});
test("embeddingSize setter stores the value and recomputes weightMatrixColumns", () => {
    const hyperparameters = new Hyperparameters(768, 1024, 0.1, Hyperparameters.TRAINING_MODE, 12);
    hyperparameters.embeddingSize = 1200;
    assert.equal(hyperparameters.embeddingSize, 1200);
    assert.equal(hyperparameters.weightMatrixColumns, 100);
});
test("embeddingSize setter rejects a size not divisible by numberAttentionHeads and leaves state unchanged", () => {
    const hyperparameters = new Hyperparameters(768, 1024, 0.1, Hyperparameters.TRAINING_MODE, 12);
    assert.throws(() => { hyperparameters.embeddingSize = 770; }, RangeError);
    assert.throws(() => { hyperparameters.embeddingSize = NaN; }, RangeError);
    assert.equal(hyperparameters.embeddingSize, 768);
    assert.equal(hyperparameters.weightMatrixColumns, 64);
});
test("numberAttentionHeads setter stores the value and recomputes weightMatrixColumns", () => {
    const hyperparameters = new Hyperparameters(768, 1024, 0.1, Hyperparameters.TRAINING_MODE, 12);
    hyperparameters.numberAttentionHeads = 8;
    assert.equal(hyperparameters.numberAttentionHeads, 8);
    assert.equal(hyperparameters.weightMatrixColumns, 96);
});
test("numberAttentionHeads setter rejects invalid values and leaves state unchanged", () => {
    const hyperparameters = new Hyperparameters(768, 1024, 0.1, Hyperparameters.TRAINING_MODE, 12);
    for (const value of NON_POSITIVE_INTEGERS) {
        assert.throws(() => { hyperparameters.numberAttentionHeads = value; }, RangeError, `accepted ${value}`);
    }
    // A positive whole number that does not divide 768 exactly.
    assert.throws(() => { hyperparameters.numberAttentionHeads = 7; }, RangeError);
    assert.equal(hyperparameters.numberAttentionHeads, 12);
    assert.equal(hyperparameters.weightMatrixColumns, 64);
});
test("dropoutRate setter accepts the boundaries 0 and 1", () => {
    const hyperparameters = new Hyperparameters();
    hyperparameters.dropoutRate = 0;
    assert.equal(hyperparameters.dropoutRate, 0);
    hyperparameters.dropoutRate = 1;
    assert.equal(hyperparameters.dropoutRate, 1);
});
test("dropoutRate setter rejects values outside [0, 1] and NaN, leaving state unchanged", () => {
    const hyperparameters = new Hyperparameters();
    for (const value of [-0.1, 1.1, NaN]) {
        assert.throws(() => { hyperparameters.dropoutRate = value; }, RangeError, `accepted ${value}`);
    }
    assert.equal(hyperparameters.dropoutRate, 0.1);
});
test("numberTransformerBlocks setter stores a positive whole number", () => {
    const hyperparameters = new Hyperparameters();
    hyperparameters.numberTransformerBlocks = 24;
    assert.equal(hyperparameters.numberTransformerBlocks, 24);
});
test("numberTransformerBlocks setter rejects invalid values and leaves state unchanged", () => {
    const hyperparameters = new Hyperparameters();
    for (const value of NON_POSITIVE_INTEGERS) {
        assert.throws(() => { hyperparameters.numberTransformerBlocks = value; }, RangeError, `accepted ${value}`);
    }
    assert.equal(hyperparameters.numberTransformerBlocks, 12);
});
test("batchSize setter stores a positive whole number", () => {
    const hyperparameters = new Hyperparameters();
    hyperparameters.batchSize = 8;
    assert.equal(hyperparameters.batchSize, 8);
});
test("batchSize setter rejects invalid values and leaves state unchanged", () => {
    const hyperparameters = new Hyperparameters();
    for (const value of NON_POSITIVE_INTEGERS) {
        assert.throws(() => { hyperparameters.batchSize = value; }, RangeError, `accepted ${value}`);
    }
    assert.equal(hyperparameters.batchSize, 2);
});
test("positionEmbeddingStandardDeviation setter accepts 0 and positive finite values", () => {
    const hyperparameters = new Hyperparameters();
    hyperparameters.positionEmbeddingStandardDeviation = 0;
    assert.equal(hyperparameters.positionEmbeddingStandardDeviation, 0);
    hyperparameters.positionEmbeddingStandardDeviation = 0.02;
    assert.equal(hyperparameters.positionEmbeddingStandardDeviation, 0.02);
});
// Draws count values from a Hyperparameters' random number stream.
const draw = (hyperparameters, count) => Array.from({ length: count }, () => hyperparameters.random());
test("seed defaults to undefined", () => {
    assert.equal(new Hyperparameters().seed, undefined);
});
test("random() produces the same stream for the same seed", () => {
    const first = new Hyperparameters();
    const second = new Hyperparameters();
    first.seed = 42;
    second.seed = 42;
    assert.deepEqual(draw(first, 10), draw(second, 10));
});
test("setting the seed restarts the random stream", () => {
    const hyperparameters = new Hyperparameters();
    hyperparameters.seed = 42;
    const before = draw(hyperparameters, 10);
    hyperparameters.seed = 42;
    assert.equal(hyperparameters.seed, 42);
    assert.deepEqual(draw(hyperparameters, 10), before);
});
test("random() produces different streams for different seeds", () => {
    const first = new Hyperparameters();
    const second = new Hyperparameters();
    first.seed = 1;
    second.seed = 2;
    assert.notDeepEqual(draw(first, 10), draw(second, 10));
});
test("positionEmbeddingStandardDeviation setter rejects negative, infinite and NaN values, leaving state unchanged", () => {
    const hyperparameters = new Hyperparameters();
    for (const value of [-0.01, Infinity, NaN]) {
        assert.throws(() => { hyperparameters.positionEmbeddingStandardDeviation = value; }, RangeError, `accepted ${value}`);
    }
    assert.equal(hyperparameters.positionEmbeddingStandardDeviation, 0.01);
});
// The weight initialization standard deviations that default to GPT-2's 0.02.
const gpt2StandardDeviations = [
    "tokenEmbeddingStandardDeviation",
    "linearLayerStandardDeviation",
    "outputLayerStandardDeviation",
];
test("weight initialization standard deviations default to 0.02, as in GPT-2", () => {
    const hyperparameters = new Hyperparameters();
    for (const name of gpt2StandardDeviations) {
        assert.equal(hyperparameters[name], 0.02, name);
    }
});
test("weight initialization standard deviation setters accept 0 and positive finite values", () => {
    const hyperparameters = new Hyperparameters();
    for (const name of gpt2StandardDeviations) {
        hyperparameters[name] = 0;
        assert.equal(hyperparameters[name], 0, name);
        hyperparameters[name] = 0.05;
        assert.equal(hyperparameters[name], 0.05, name);
    }
});
test("weight initialization standard deviation setters reject negative, infinite and NaN values, leaving state unchanged", () => {
    const hyperparameters = new Hyperparameters();
    for (const name of gpt2StandardDeviations) {
        for (const value of [-0.01, Infinity, NaN]) {
            assert.throws(() => { hyperparameters[name] = value; }, RangeError, `${name} accepted ${value}`);
        }
        assert.equal(hyperparameters[name], 0.02, name);
    }
});
