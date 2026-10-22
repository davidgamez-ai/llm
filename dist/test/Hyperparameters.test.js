import { test } from "node:test";
import assert from "node:assert/strict";
import Hyperparameters from "../Hyperparameters.js";
import GPTModel from "../GPTModel.js";
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
    assert.equal(hyperparameters.contextLength, 1024);
});
test("forReducedTraining() defaults to 2.5 hours: a small model for one pass over about 2,200 stories", () => {
    const hyperparameters = Hyperparameters.forReducedTraining();
    assert.equal(hyperparameters.embeddingSize, 64);
    assert.equal(hyperparameters.numberAttentionHeads, 4);
    assert.equal(hyperparameters.weightMatrixColumns, 16);
    assert.equal(hyperparameters.numberTransformerBlocks, 4);
    assert.equal(hyperparameters.contextLength, 128);
    assert.equal(hyperparameters.dropoutRate, 0);
    assert.equal(hyperparameters.learningRate, 0.002);
    assert.equal(hyperparameters.numberEpochs, 1);
    // 2.5 hours x 187,000 tokens an hour / 210 tokens a story.
    assert.equal(hyperparameters.numberTrainingStories, 2226);
    assert.equal(hyperparameters.outputLength, 50);
    // Unchanged from the defaults.
    assert.equal(hyperparameters.vocabularySize, 50257);
    assert.equal(hyperparameters.numberValidationStories, 100);
    assert.equal(hyperparameters.qkvBias, true);
    assert.equal(hyperparameters.mode, Hyperparameters.TRAINING_MODE);
});
test("forReducedTraining(hours) trains on stories in proportion to the hours", () => {
    assert.equal(Hyperparameters.forReducedTraining(1).numberTrainingStories, 890);
    assert.equal(Hyperparameters.forReducedTraining(5).numberTrainingStories, 4452);
    assert.equal(Hyperparameters.forReducedTraining(10).numberTrainingStories, 8905);
});
test("forReducedTraining(hours) keeps the narrow model below 20 hours and uses the wider one from 20 hours", () => {
    const below = Hyperparameters.forReducedTraining(19.9);
    const from = Hyperparameters.forReducedTraining(20);
    assert.equal(below.embeddingSize, 64);
    assert.equal(below.learningRate, 0.002);
    assert.equal(from.embeddingSize, 128);
    assert.equal(from.weightMatrixColumns, 32);
    assert.equal(from.learningRate, 0.001);
    // The wider model gets through half as many tokens an hour: 20 x 93,500 / 210.
    assert.equal(from.numberTrainingStories, 8905);
});
test("forReducedTraining(hours) trains on at least 10 stories, however short the time", () => {
    assert.equal(Hyperparameters.forReducedTraining(0.001).numberTrainingStories, 10);
});
test("forReducedTraining(hours) rejects hours that are not a finite number greater than 0", () => {
    for (const hours of [0, -1, NaN, Infinity]) {
        assert.throws(() => Hyperparameters.forReducedTraining(hours), RangeError, `accepted ${hours}`);
    }
});
test("forReducedTraining() returns a new instance each time, leaving the GPT-2 small defaults alone", () => {
    const first = Hyperparameters.forReducedTraining();
    first.numberTrainingStories = 10;
    assert.equal(Hyperparameters.forReducedTraining().numberTrainingStories, 2226);
    assert.equal(new Hyperparameters().embeddingSize, 768);
});
test("forReducedTraining() builds a model of about 6.5 million parameters, almost all in the vocabulary", () => {
    const model = new GPTModel(Hyperparameters.forReducedTraining());
    const total = model.getParameterCount();
    const vocabulary = 2 * 50257 * 64;
    assert.ok(total > 6.4e6 && total < 6.8e6, `${total} parameters`);
    // The token embedding and output layer: 50,257 x 64 each.
    assert.ok(vocabulary / total > 0.95);
});
test("the architecture defaults are those of GPT-2 small", () => {
    const hyperparameters = new Hyperparameters();
    assert.equal(hyperparameters.vocabularySize, 50257);
    assert.equal(hyperparameters.embeddingSize, 768);
    assert.equal(hyperparameters.contextLength, 1024);
    assert.equal(hyperparameters.numberAttentionHeads, 12);
    assert.equal(hyperparameters.numberTransformerBlocks, 12);
    assert.equal(hyperparameters.dropoutRate, 0.1);
    assert.equal(hyperparameters.qkvBias, true);
    assert.equal(hyperparameters.feedForwardBias, true);
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
test("the optimizer hyperparameters have the book's defaults", () => {
    const hyperparameters = new Hyperparameters();
    assert.equal(hyperparameters.learningRate, 0.0004);
    assert.equal(hyperparameters.beta1, 0.9);
    assert.equal(hyperparameters.beta2, 0.999);
    assert.equal(hyperparameters.optimizerEpsilon, 1e-8);
    assert.equal(hyperparameters.weightDecay, 0.1);
});
test("the optimizer hyperparameter setters store valid values", () => {
    const hyperparameters = new Hyperparameters();
    hyperparameters.learningRate = 0.001;
    hyperparameters.beta1 = 0;
    hyperparameters.beta2 = 0.99;
    hyperparameters.optimizerEpsilon = 1e-6;
    hyperparameters.weightDecay = 0;
    assert.equal(hyperparameters.learningRate, 0.001);
    assert.equal(hyperparameters.beta1, 0);
    assert.equal(hyperparameters.beta2, 0.99);
    assert.equal(hyperparameters.optimizerEpsilon, 1e-6);
    assert.equal(hyperparameters.weightDecay, 0);
});
test("the optimizer hyperparameter setters reject invalid values, leaving state unchanged", () => {
    const hyperparameters = new Hyperparameters();
    for (const value of [0, -0.001, Infinity, NaN]) {
        assert.throws(() => { hyperparameters.learningRate = value; }, RangeError, `learningRate accepted ${value}`);
        assert.throws(() => { hyperparameters.optimizerEpsilon = value; }, RangeError, `optimizerEpsilon accepted ${value}`);
    }
    for (const value of [-0.1, 1, 1.5, NaN]) {
        assert.throws(() => { hyperparameters.beta1 = value; }, RangeError, `beta1 accepted ${value}`);
        assert.throws(() => { hyperparameters.beta2 = value; }, RangeError, `beta2 accepted ${value}`);
    }
    for (const value of [-0.1, Infinity, NaN]) {
        assert.throws(() => { hyperparameters.weightDecay = value; }, RangeError, `weightDecay accepted ${value}`);
    }
    assert.equal(hyperparameters.learningRate, 0.0004);
    assert.equal(hyperparameters.beta1, 0.9);
    assert.equal(hyperparameters.beta2, 0.999);
    assert.equal(hyperparameters.optimizerEpsilon, 1e-8);
    assert.equal(hyperparameters.weightDecay, 0.1);
});
test("numberTrainingStories defaults to 1000 and numberValidationStories to 100", () => {
    const hyperparameters = new Hyperparameters();
    assert.equal(hyperparameters.numberTrainingStories, 1000);
    assert.equal(hyperparameters.numberValidationStories, 100);
});
test("the story count setters store positive whole numbers", () => {
    const hyperparameters = new Hyperparameters();
    hyperparameters.numberTrainingStories = 5000;
    hyperparameters.numberValidationStories = 1;
    assert.equal(hyperparameters.numberTrainingStories, 5000);
    assert.equal(hyperparameters.numberValidationStories, 1);
});
test("the story count setters reject invalid values, leaving state unchanged", () => {
    const hyperparameters = new Hyperparameters();
    for (const value of NON_POSITIVE_INTEGERS) {
        assert.throws(() => { hyperparameters.numberTrainingStories = value; }, RangeError, `numberTrainingStories accepted ${value}`);
        assert.throws(() => { hyperparameters.numberValidationStories = value; }, RangeError, `numberValidationStories accepted ${value}`);
    }
    assert.equal(hyperparameters.numberTrainingStories, 1000);
    assert.equal(hyperparameters.numberValidationStories, 100);
});
test("the constructor rejects invalid story counts", () => {
    // Arguments 22 and 23 are numberTrainingStories and numberValidationStories.
    const build = (numberTrainingStories, numberValidationStories) => new Hyperparameters(768, 12, 0.1, Hyperparameters.TRAINING_MODE, 12, 12, 2, false, true, 0.01, 0.02, 0.02, 0.02, 2, 0.9, 1, 0.0004, 0.9, 0.999, 1e-8, 0.1, numberTrainingStories, numberValidationStories);
    assert.doesNotThrow(() => build(10, 2));
    assert.throws(() => build(0, 2), RangeError);
    assert.throws(() => build(10, 1.5), RangeError);
});
test("the constructor rejects invalid optimizer hyperparameters", () => {
    // Arguments 17 to 21 are learningRate, beta1, beta2, optimizerEpsilon and weightDecay.
    const build = (learningRate, beta1, beta2, optimizerEpsilon, weightDecay) => new Hyperparameters(768, 12, 0.1, Hyperparameters.TRAINING_MODE, 12, 12, 2, false, true, 0.01, 0.02, 0.02, 0.02, 2, 0.9, 1, learningRate, beta1, beta2, optimizerEpsilon, weightDecay);
    assert.doesNotThrow(() => build(0.0004, 0.9, 0.999, 1e-8, 0.1));
    assert.throws(() => build(0, 0.9, 0.999, 1e-8, 0.1), RangeError);
    assert.throws(() => build(0.0004, 1, 0.999, 1e-8, 0.1), RangeError);
    assert.throws(() => build(0.0004, 0.9, -0.1, 1e-8, 0.1), RangeError);
    assert.throws(() => build(0.0004, 0.9, 0.999, 0, 0.1), RangeError);
    assert.throws(() => build(0.0004, 0.9, 0.999, 1e-8, -1), RangeError);
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
