import { test } from "node:test";
import assert from "node:assert/strict";
import Hyperparameters from "../Hyperparameters.js";
import TrainingPlanner from "../TrainingPlanner.js";
// The number of parameters in each model size.
const SMALL = 13675520;
const MEDIUM = 49300000;
const LARGE = 163037184;
const DATASETS = ["verdict", "shakespeare", "tinyStories", "alpaca"];
test("pre-training uses the learning rate of the model's size, on every dataset", () => {
    for (const dataset of DATASETS) {
        assert.equal(TrainingPlanner.plan("pretrain", dataset, 1, Hyperparameters.smallModel(), SMALL).learningRate, 0.001);
        assert.equal(TrainingPlanner.plan("pretrain", dataset, 1, Hyperparameters.mediumModel(), MEDIUM).learningRate, 0.0005);
        assert.equal(TrainingPlanner.plan("pretrain", dataset, 1, Hyperparameters.largeModel(), LARGE).learningRate, 0.0004);
    }
});
test("pre-training's learning rate does not depend on one left in the hyperparameters by fine tuning", () => {
    const hyperparameters = Hyperparameters.smallModel();
    hyperparameters.learningRate = TrainingPlanner.FINE_TUNING_LEARNING_RATE;
    assert.equal(TrainingPlanner.plan("pretrain", "tinyStories", 1, hyperparameters, SMALL).learningRate, 0.001);
});
test("fine tuning uses the same low learning rate and larger batches for every model size", () => {
    for (const dataset of DATASETS) {
        for (const [hyperparameters, parameters] of [[Hyperparameters.smallModel(), SMALL], [Hyperparameters.largeModel(), LARGE]]) {
            const plan = TrainingPlanner.plan("fineTune", dataset, 10, hyperparameters, parameters);
            assert.equal(plan.learningRate, 0.00005);
            assert.equal(plan.batchSize, 8);
        }
    }
});
test("windows are as long as the model's context, up to 256 tokens", () => {
    assert.equal(TrainingPlanner.plan("pretrain", "verdict", 1, Hyperparameters.smallModel(), SMALL).trainingWindowLength, 128);
    assert.equal(TrainingPlanner.plan("pretrain", "verdict", 1, Hyperparameters.mediumModel(), MEDIUM).trainingWindowLength, 256);
    assert.equal(TrainingPlanner.plan("pretrain", "verdict", 1, Hyperparameters.largeModel(), LARGE).trainingWindowLength, 256);
});
test("TinyStories fills the hours, after allowing for validation, using the model's number of parameters", () => {
    // 10 x 1.35 trillion / 13,675,520 parameters = 987,165 tokens, less 100 x 210 / 3 for validation, over 210 tokens a story.
    assert.equal(TrainingPlanner.plan("pretrain", "tinyStories", 10, Hyperparameters.smallModel(), SMALL).numberTrainingExamples, 4667);
    // 50 x 1.35 trillion / 163,037,184 parameters = 414,017 tokens: far fewer tokens an hour than the small model.
    assert.equal(TrainingPlanner.plan("pretrain", "tinyStories", 50, Hyperparameters.largeModel(), LARGE).numberTrainingExamples, 1938);
    // The training speed can be given: twice as fast, about twice the stories.
    assert.equal(TrainingPlanner.plan("pretrain", "tinyStories", 10, Hyperparameters.smallModel(), SMALL, 2.7e12).numberTrainingExamples, 9368);
});
test("TinyStories and Alpaca use one epoch, the whole of each example and an estimate close to the hours", () => {
    for (const dataset of ["tinyStories", "alpaca"]) {
        const plan = TrainingPlanner.plan("pretrain", dataset, 10, Hyperparameters.smallModel(), SMALL);
        assert.equal(plan.numberEpochs, 1);
        assert.equal(plan.maximumCharacters, undefined);
        assert.ok(Math.abs(plan.estimatedHours - 10) < 0.01, `${dataset} estimated ${plan.estimatedHours}`);
    }
});
test("Alpaca's examples are counted at the average length of those that fit in the window", () => {
    // 10 hours on the small model is 987,165 tokens, less 100 x 83 / 3 for validation, over 83 tokens an example.
    assert.equal(TrainingPlanner.plan("pretrain", "alpaca", 10, Hyperparameters.smallModel(), SMALL).numberTrainingExamples, 11860);
    // 50 hours on the large model is 414,017 tokens, less 100 x 106 / 3 for validation, over 106 tokens an example.
    assert.equal(TrainingPlanner.plan("fineTune", "alpaca", 50, Hyperparameters.largeModel(), LARGE).numberTrainingExamples, 3872);
});
test("Alpaca never asks for more examples than can be used, and the estimate shows the run will be short", () => {
    const plan = TrainingPlanner.plan("pretrain", "alpaca", 100, Hyperparameters.smallModel(), SMALL);
    // 36,826 examples fit in the small model's window, less 100 for validation.
    assert.equal(plan.numberTrainingExamples, 36726);
    assert.ok(plan.estimatedHours < 50, `estimated ${plan.estimatedHours}`);
});
test("a very short time still gives at least 10 examples, and at least enough for two batches", () => {
    assert.equal(TrainingPlanner.plan("pretrain", "tinyStories", 0.001, Hyperparameters.smallModel(), SMALL).numberTrainingExamples, 10);
    // Two batches of 8 windows of 256 tokens are 4,096 tokens, or 20 stories of 210 tokens.
    const fineTuning = TrainingPlanner.plan("fineTune", "tinyStories", 0.001, Hyperparameters.largeModel(), LARGE);
    assert.equal(fineTuning.numberTrainingExamples, 20);
    assert.ok(fineTuning.estimatedHours > 0.001);
});
test("a short time uses only the start of a text, for one epoch", () => {
    // 1 hour on the large model is 8,280 tokens, far less than Tiny Shakespeare's 301,966.
    const plan = TrainingPlanner.plan("pretrain", "shakespeare", 1, Hyperparameters.largeModel(), LARGE);
    assert.equal(plan.numberEpochs, 1);
    assert.equal(plan.numberTrainingExamples, undefined);
    assert.ok(plan.maximumCharacters !== undefined && plan.maximumCharacters > 20000 && plan.maximumCharacters < 35000, `${plan.maximumCharacters} characters`);
    assert.ok(Math.abs(plan.estimatedHours - 1) < 0.01, `estimated ${plan.estimatedHours}`);
});
test("a long time repeats the whole text for as many epochs as fit", () => {
    // 10 hours on the small model is 987,165 tokens, about 3 passes of Tiny Shakespeare.
    const plan = TrainingPlanner.plan("pretrain", "shakespeare", 10, Hyperparameters.smallModel(), SMALL);
    assert.equal(plan.maximumCharacters, undefined);
    assert.equal(plan.numberEpochs, 3);
    assert.ok(plan.estimatedHours > 8 && plan.estimatedHours <= 10, `estimated ${plan.estimatedHours}`);
});
test("passes over a text are limited, more tightly for fine tuning, and the estimate shows the run will be short", () => {
    const pretraining = TrainingPlanner.plan("pretrain", "verdict", 100, Hyperparameters.smallModel(), SMALL);
    const fineTuning = TrainingPlanner.plan("fineTune", "verdict", 100, Hyperparameters.smallModel(), SMALL);
    assert.equal(pretraining.numberEpochs, 50);
    assert.equal(fineTuning.numberEpochs, 10);
    assert.ok(fineTuning.estimatedHours < pretraining.estimatedHours);
    assert.ok(pretraining.estimatedHours < 100);
});
test("a very short time on a text still leaves enough for two batches and a validation window", () => {
    for (const kind of ["pretrain", "fineTune"]) {
        for (const dataset of ["verdict", "shakespeare"]) {
            for (const [hyperparameters, parameters] of [[Hyperparameters.smallModel(), SMALL], [Hyperparameters.largeModel(), LARGE]]) {
                const plan = TrainingPlanner.plan(kind, dataset, 0.001, hyperparameters, parameters);
                const characters = plan.maximumCharacters ?? (dataset === "verdict" ? 20643 : 1115394);
                const charactersPerToken = dataset === "verdict" ? 20643 / 5310 : 1115394 / 338025;
                const tokens = characters / charactersPerToken;
                // 90% of the tokens for training, in two batches; 10% for validation, in one window.
                const description = `${kind} on ${dataset} with windows of ${plan.trainingWindowLength}: ${tokens} tokens`;
                assert.ok(0.9 * tokens >= 2 * plan.batchSize * plan.trainingWindowLength, description);
                assert.ok(0.1 * tokens >= plan.trainingWindowLength + 1, description);
            }
        }
    }
});
