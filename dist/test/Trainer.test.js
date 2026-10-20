import { test } from "node:test";
import assert from "node:assert/strict";
import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import Hyperparameters from "../Hyperparameters.js";
import GPTModel from "../GPTModel.js";
import Trainer from "../Trainer.js";
import BPETokenizer from "../BPETokenizer.js";
/*
 * Writes text to a new file in a temporary folder and returns its path, so
 * the tests do not depend on the data folder or the working directory.
 */
const writeTempFile = (text) => {
    const folder = fs.mkdtempSync(path.join(os.tmpdir(), "trainer-test-"));
    const filePath = path.join(folder, "data.txt");
    fs.writeFileSync(filePath, text, "utf-8");
    return filePath;
};
// About 2,000 tokens of varied text: enough for many windows in both sets.
const TEXT = Array.from({ length: 200 }, (_, i) => `Sentence ${i} is about item ${(i * 7) % 13}.`).join(" ");
/*
 * Builds a tiny model (embeddingSize 8, 2 heads, 1 block) so the tests run
 * quickly, with contextLength 4 and batchSize 3. It starts in
 * GENERATION_MODE so the tests can check the mode is restored. Every call to
 * the model's update() is recorded, with the mode it was called in.
 */
const setUp = (numberEpochs = 1) => {
    const hyperparameters = new Hyperparameters(8, 4, 0.1, Hyperparameters.GENERATION_MODE, 2, 1, 3);
    hyperparameters.numberEpochs = numberEpochs;
    hyperparameters.seed = 42;
    const model = new GPTModel(hyperparameters);
    const calls = [];
    const update = model.update.bind(model);
    model.update = (inputs) => {
        calls.push({ inputs: [...inputs], mode: hyperparameters.mode });
        return update(inputs);
    };
    const trainer = new Trainer(hyperparameters, model);
    trainer.load(writeTempFile(TEXT));
    return { hyperparameters, trainer, calls };
};
test("load() splits the text 90/10 by characters before tokenizing", () => {
    const { trainer, calls } = setUp();
    const tokenizer = new BPETokenizer();
    const splitIndex = Math.floor(0.9 * TEXT.length);
    const trainingTokenCount = tokenizer.encode(TEXT.slice(0, splitIndex)).length;
    const validationTokenCount = tokenizer.encode(TEXT.slice(splitIndex)).length;
    trainer.train();
    // Full windows of 4 inputs need 5 tokens, then one more token for each further window.
    const trainingWindows = Math.floor((trainingTokenCount - 1) / 4);
    const validationWindows = Math.floor((validationTokenCount - 1) / 4);
    const trainingCalls = calls.filter((call) => call.mode === Hyperparameters.TRAINING_MODE).length;
    const validationCalls = calls.filter((call) => call.mode === Hyperparameters.VALIDATION_MODE).length;
    // Training drops the incomplete last batch of 3; validation keeps every window.
    assert.equal(trainingCalls, Math.floor(trainingWindows / 3) * 3);
    assert.equal(validationCalls, validationWindows);
});
test("train() feeds only full windows of contextLength tokens", () => {
    const { trainer, calls } = setUp();
    trainer.train();
    assert.ok(calls.length > 0);
    assert.ok(calls.every((call) => call.inputs.length === 4));
});
test("train() returns finite losses for every epoch and restores the original mode", () => {
    const { hyperparameters, trainer } = setUp(2);
    const losses = trainer.train();
    assert.deepEqual(losses.map((loss) => loss.epoch), [1, 2]);
    assert.ok(losses.every((loss) => Number.isFinite(loss.trainingLoss) && Number.isFinite(loss.validationLoss)));
    assert.equal(hyperparameters.mode, Hyperparameters.GENERATION_MODE);
});
test("train() shuffles the training windows differently in each epoch", () => {
    const { trainer, calls } = setUp(2);
    trainer.train();
    const training = calls.filter((call) => call.mode === Hyperparameters.TRAINING_MODE).map((call) => JSON.stringify(call.inputs));
    const firstEpoch = training.slice(0, training.length / 2);
    const secondEpoch = training.slice(training.length / 2);
    assert.notDeepEqual(firstEpoch, secondEpoch);
});
test("calculateValidationLoss() is deterministic and matches the loss reported by train()", () => {
    const { trainer } = setUp();
    const losses = trainer.train();
    // Validation mode has no dropout and no shuffling, so it never changes while the weights stay the same.
    assert.equal(trainer.calculateValidationLoss(), losses[0].validationLoss);
    assert.equal(trainer.calculateValidationLoss(), losses[0].validationLoss);
});
test("calculateValidationLoss() of the untrained model matches the recorded reference", () => {
    const { trainer } = setUp();
    /*
     * Recorded before the loss calculation was moved into Loss.ts, so this
     * catches any change to the loss or the forward pass. Close to
     * ln(vocabularySize) = ln(50257), about 10.82, as expected for an untrained model.
     */
    assert.ok(Math.abs(trainer.calculateValidationLoss() - 10.81001060108122) <= 1e-12);
});
test("train() throws a clear error when there is not enough text for one batch", () => {
    const hyperparameters = new Hyperparameters(8, 4, 0.1, Hyperparameters.TRAINING_MODE, 2, 1, 3);
    const trainer = new Trainer(hyperparameters, new GPTModel(hyperparameters));
    trainer.load(writeTempFile("Far too short."));
    assert.throws(() => trainer.train(), /at least batchSize/);
});
