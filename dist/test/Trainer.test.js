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
/** Builds varied text from count numbered sentences. */
const sentences = (count) => Array.from({ length: count }, (_, i) => `Sentence ${i} is about item ${(i * 7) % 13}.`).join(" ");
/*
 * About 2,000 tokens of varied text. Only used by the reference test, whose
 * recorded validation loss depends on it.
 */
const REFERENCE_TEXT = sentences(200);
/*
 * About 400 tokens: enough for many windows in both sets. Training now
 * runs backpropagation and weight updates for every window, so the tests
 * that call train() use this shorter text to stay quick.
 */
const TEXT = sentences(40);
/*
 * Builds a tiny model (embeddingSize 8, 2 heads, 1 block) so the tests run
 * quickly, with contextLength 4 and batchSize 3. It starts in
 * GENERATION_MODE so the tests can check the mode is restored. Every call to
 * the model's update(), calculateGradients() and updateWeights() is
 * recorded, with the mode update() was called in.
 */
const setUp = (numberEpochs = 1, text = TEXT) => {
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
    // The sequence of training calls: "average" or "replace" for calculateGradients(), "update" for updateWeights().
    const trainingCalls = [];
    const calculateGradients = model.calculateGradients.bind(model);
    model.calculateGradients = (batchAverage) => {
        trainingCalls.push(batchAverage ? "average" : "replace");
        calculateGradients(batchAverage);
    };
    const updateWeights = model.updateWeights.bind(model);
    model.updateWeights = (optimizer) => {
        trainingCalls.push("update");
        updateWeights(optimizer);
    };
    const trainer = new Trainer(hyperparameters, model);
    trainer.load(writeTempFile(text));
    return { hyperparameters, model, trainer, calls, trainingCalls };
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
test("for each batch, train() replaces the gradients with the first window, averages in the rest, then updates the weights once", () => {
    const { trainer, calls, trainingCalls } = setUp();
    trainer.train();
    // With batchSize 3, every batch is: replace, average, average, update.
    const batchCount = calls.filter((call) => call.mode === Hyperparameters.TRAINING_MODE).length / 3;
    const expected = Array.from({ length: batchCount }, () => ["replace", "average", "average", "update"]).flat();
    assert.ok(batchCount > 1);
    assert.deepEqual(trainingCalls, expected);
});
test("train() changes the weights, so the validation loss changes from epoch to epoch", () => {
    const { model, trainer } = setUp(2);
    const before = Float32Array.from(model.linearOutputLayer.weights.data);
    const losses = trainer.train();
    assert.ok(model.linearOutputLayer.weights.data.some((value, i) => value !== before[i]));
    assert.notEqual(losses[0].validationLoss, losses[1].validationLoss);
});
test("calculateValidationLoss() is deterministic and matches the loss reported at the end of training", () => {
    const { trainer } = setUp();
    const losses = trainer.train();
    // Validation mode has no dropout and no shuffling, so it never changes while the weights stay the same.
    assert.equal(trainer.calculateValidationLoss(), losses[0].validationLoss);
    assert.equal(trainer.calculateValidationLoss(), losses[0].validationLoss);
});
test("calculateValidationLoss() of the untrained model matches the recorded reference", () => {
    const { trainer } = setUp(1, REFERENCE_TEXT);
    /*
     * Recorded before the loss calculation was moved into Loss.ts, so this
     * catches any change to the loss or the forward pass. Close to
     * ln(vocabularySize) = ln(50257), about 10.82, as expected for an untrained model.
     */
    assert.ok(Math.abs(trainer.calculateValidationLoss() - 10.81001060108122) <= 1e-12);
});
test("training on short, repetitive text drives the loss far below that of an untrained model", () => {
    /*
     * The text repeats two sentences, so a model that learns should soon
     * predict it well: it overfits. A higher learning rate than the default
     * (0.02 rather than 0.0004) makes this happen in a few epochs. The loss of
     * an untrained model is about ln(50257) = 10.8.
     */
    const { hyperparameters, trainer } = setUp(8, "The cat sat on the mat. The dog sat on the log. ".repeat(6));
    hyperparameters.learningRate = 0.02;
    const losses = trainer.train();
    const first = losses[0].trainingLoss;
    const last = losses[losses.length - 1].trainingLoss;
    assert.ok(last < 3, `final training loss ${last}`);
    assert.ok(last < first / 3, `training loss fell only from ${first} to ${last}`);
    assert.ok(losses[losses.length - 1].validationLoss < 3, `final validation loss ${losses[losses.length - 1].validationLoss}`);
});
/** Reads the trainer's private token arrays, to check what load() produced. */
const loadedTokens = (trainer) => trainer;
/** The tokens of the training and validation parts of one text, split 90/10 by characters. */
const split = (text) => {
    const tokenizer = new BPETokenizer();
    const splitIndex = Math.floor(0.9 * text.length);
    return { training: tokenizer.encode(text.slice(0, splitIndex)), validation: tokenizer.encode(text.slice(splitIndex)) };
};
test("load() with several files splits each file 90/10 and joins the parts with <|endoftext|>", () => {
    const first = sentences(30);
    const second = "A different book about other things, which goes on for a while. ".repeat(20);
    const hyperparameters = new Hyperparameters(8, 4, 0.1, Hyperparameters.TRAINING_MODE, 2, 1, 3);
    const trainer = new Trainer(hyperparameters, new GPTModel(hyperparameters));
    const endOfText = new BPETokenizer().endOfTextId;
    trainer.load([writeTempFile(first), writeTempFile(second)]);
    // Each set holds the matching part of every file, in order, separated by the end-of-text token.
    const a = split(first);
    const b = split(second);
    assert.deepEqual(loadedTokens(trainer).trainingTokens, [...a.training, endOfText, ...b.training]);
    assert.deepEqual(loadedTokens(trainer).validationTokens, [...a.validation, endOfText, ...b.validation]);
});
test("load() gives the same tokens for one path as for an array holding that path", () => {
    const filePath = writeTempFile(TEXT);
    const hyperparameters = new Hyperparameters(8, 4, 0.1, Hyperparameters.TRAINING_MODE, 2, 1, 3);
    const single = new Trainer(hyperparameters, new GPTModel(hyperparameters));
    const array = new Trainer(hyperparameters, new GPTModel(hyperparameters));
    single.load(filePath);
    array.load([filePath]);
    assert.deepEqual(loadedTokens(array).trainingTokens, loadedTokens(single).trainingTokens);
    assert.deepEqual(loadedTokens(array).validationTokens, loadedTokens(single).validationTokens);
    // And no end-of-text token is added for a single file.
    assert.deepEqual(loadedTokens(single).trainingTokens, split(TEXT).training);
});
test("load() replaces previously loaded tokens", () => {
    const { trainer } = setUp();
    trainer.load(writeTempFile(sentences(10)));
    assert.deepEqual(loadedTokens(trainer).trainingTokens, split(sentences(10)).training);
});
test("load() rejects an empty array of files", () => {
    const hyperparameters = new Hyperparameters(8, 4, 0.1, Hyperparameters.TRAINING_MODE, 2, 1, 3);
    const trainer = new Trainer(hyperparameters, new GPTModel(hyperparameters));
    assert.throws(() => trainer.load([]), /at least one file/);
});
/** Writes a CSV file in the TinyStories format: a "text" column with one quoted story per record. */
const writeStoriesCsv = (stories) => writeTempFile("text\n" + stories.map((story) => `"${story.replace(/"/g, '""')}"`).join("\n") + "\n");
/** Builds count numbered stories, each containing a quote and a line break as real TinyStories do. */
const makeStories = (count, label) => Array.from({ length: count }, (_, i) => `${label} story ${i}. Tom said, "Hello ${i}!"\n\nThe end.`);
test("loadTinyStories() loads the first numberTrainingStories and numberValidationStories, joined with <|endoftext|>", () => {
    const hyperparameters = new Hyperparameters(8, 4, 0.1, Hyperparameters.TRAINING_MODE, 2, 1, 3);
    hyperparameters.numberTrainingStories = 3;
    hyperparameters.numberValidationStories = 2;
    const trainer = new Trainer(hyperparameters, new GPTModel(hyperparameters));
    const trainingStories = makeStories(5, "Training");
    const validationStories = makeStories(4, "Validation");
    const tokenizer = new BPETokenizer();
    const endOfText = tokenizer.endOfTextId;
    trainer.loadTinyStories(writeStoriesCsv(trainingStories), writeStoriesCsv(validationStories));
    // Only the first 3 and 2 stories, each encoded separately, with the end-of-text token between them.
    const [t0, t1, t2] = trainingStories.map((story) => tokenizer.encode(story));
    const [v0, v1] = validationStories.map((story) => tokenizer.encode(story));
    assert.deepEqual(loadedTokens(trainer).trainingTokens, [...t0, endOfText, ...t1, endOfText, ...t2]);
    assert.deepEqual(loadedTokens(trainer).validationTokens, [...v0, endOfText, ...v1]);
});
test("each loadTinyStories() call continues with the next training stories, but always uses the same validation stories", () => {
    const hyperparameters = new Hyperparameters(8, 4, 0.1, Hyperparameters.TRAINING_MODE, 2, 1, 3);
    hyperparameters.numberTrainingStories = 2;
    hyperparameters.numberValidationStories = 1;
    const trainer = new Trainer(hyperparameters, new GPTModel(hyperparameters));
    const trainingStories = makeStories(6, "Training");
    const trainingFile = writeStoriesCsv(trainingStories);
    const validationFile = writeStoriesCsv(makeStories(3, "Validation"));
    const tokenizer = new BPETokenizer();
    const endOfText = tokenizer.endOfTextId;
    const [t0, t1, t2, t3] = trainingStories.map((story) => tokenizer.encode(story));
    assert.equal(trainer.tinyStoriesLoaded, 0);
    trainer.loadTinyStories(trainingFile, validationFile);
    const firstValidation = [...loadedTokens(trainer).validationTokens];
    assert.deepEqual(loadedTokens(trainer).trainingTokens, [...t0, endOfText, ...t1]);
    assert.equal(trainer.tinyStoriesLoaded, 2);
    trainer.loadTinyStories(trainingFile, validationFile);
    assert.deepEqual(loadedTokens(trainer).trainingTokens, [...t2, endOfText, ...t3]);
    assert.deepEqual(loadedTokens(trainer).validationTokens, firstValidation);
    assert.equal(trainer.tinyStoriesLoaded, 4);
});
test("loadTinyStories() warns when the training file runs out of new stories", (t) => {
    const warnings = [];
    t.mock.method(console, "warn", (message) => warnings.push(message));
    const hyperparameters = new Hyperparameters(8, 4, 0.1, Hyperparameters.TRAINING_MODE, 2, 1, 3);
    hyperparameters.numberTrainingStories = 2;
    hyperparameters.numberValidationStories = 1;
    const trainer = new Trainer(hyperparameters, new GPTModel(hyperparameters));
    const trainingFile = writeStoriesCsv(makeStories(3, "Training"));
    const validationFile = writeStoriesCsv(makeStories(1, "Validation"));
    trainer.loadTinyStories(trainingFile, validationFile);
    trainer.loadTinyStories(trainingFile, validationFile);
    // Only story 3 was left for the second call.
    assert.equal(trainer.tinyStoriesLoaded, 3);
    assert.equal(warnings.length, 1);
    assert.match(warnings[0], /only 1 stories after the first 2; 2 were requested/);
});
test("loadTinyStories() uses every story, with a warning, when a file has fewer than requested", (t) => {
    const warnings = [];
    t.mock.method(console, "warn", (message) => warnings.push(message));
    const hyperparameters = new Hyperparameters(8, 4, 0.1, Hyperparameters.TRAINING_MODE, 2, 1, 3);
    hyperparameters.numberTrainingStories = 10;
    hyperparameters.numberValidationStories = 1;
    const trainer = new Trainer(hyperparameters, new GPTModel(hyperparameters));
    const endOfText = new BPETokenizer().endOfTextId;
    trainer.loadTinyStories(writeStoriesCsv(makeStories(2, "Training")), writeStoriesCsv(makeStories(3, "Validation")));
    // Two training stories means exactly one end-of-text token; one validation story means none.
    assert.equal(loadedTokens(trainer).trainingTokens.filter((token) => token === endOfText).length, 1);
    assert.equal(loadedTokens(trainer).validationTokens.filter((token) => token === endOfText).length, 0);
    assert.equal(warnings.length, 1);
    assert.match(warnings[0], /only 2 stories; 10 were requested/);
});
test("after loadTinyStories(), train() trains on the stories", () => {
    const hyperparameters = new Hyperparameters(8, 4, 0.1, Hyperparameters.GENERATION_MODE, 2, 1, 3);
    hyperparameters.numberTrainingStories = 6;
    hyperparameters.numberValidationStories = 2;
    hyperparameters.seed = 42;
    const trainer = new Trainer(hyperparameters, new GPTModel(hyperparameters));
    trainer.loadTinyStories(writeStoriesCsv(makeStories(6, "Training")), writeStoriesCsv(makeStories(2, "Validation")));
    const losses = trainer.train();
    assert.equal(losses.length, 1);
    assert.ok(Number.isFinite(losses[0].trainingLoss) && Number.isFinite(losses[0].validationLoss));
});
test("train() throws a clear error when there is not enough text for one batch", () => {
    const hyperparameters = new Hyperparameters(8, 4, 0.1, Hyperparameters.TRAINING_MODE, 2, 1, 3);
    const trainer = new Trainer(hyperparameters, new GPTModel(hyperparameters));
    trainer.load(writeTempFile("Far too short."));
    assert.throws(() => trainer.train(), /at least batchSize/);
});
