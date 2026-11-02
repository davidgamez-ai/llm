import { test } from "node:test";
import assert from "node:assert/strict";
import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import Hyperparameters from "../Hyperparameters.js";
import GPTModel from "../GPTModel.js";
import Trainer from "../Trainer.js";
import BPETokenizer from "../BPETokenizer.js";
import { Dataset, Window, createWindows } from "../Dataset.js";
import TextDataset from "../TextDataset.js";
import { checkGradients } from "./GradientCheck.js";

/*
 * Writes text to a new file in a temporary folder and returns its path, so
 * the tests do not depend on the data folder or the working directory.
 */
const writeTempFile = (text: string): string => {
  const folder: string = fs.mkdtempSync(path.join(os.tmpdir(), "trainer-test-"));
  const filePath: string = path.join(folder, "data.txt");
  fs.writeFileSync(filePath, text, "utf-8");
  return filePath;
};

/** Builds varied text from count numbered sentences. */
const sentences = (count: number): string =>
  Array.from({ length: count }, (_, i) => `Sentence ${i} is about item ${(i * 7) % 13}.`).join(" ");

/*
 * About 2,000 tokens of varied text. Only used by the reference test, whose
 * recorded validation loss depends on it.
 */
const REFERENCE_TEXT: string = sentences(200);

/*
 * About 400 tokens: enough for many windows in both sets. Training now
 * runs backpropagation and weight updates for every window, so the tests
 * that call train() use this shorter text to stay quick.
 */
const TEXT: string = sentences(40);

/*
 * Builds a tiny model (embeddingSize 8, 2 heads, 1 block) so the tests run
 * quickly, with contextLength 4 and batchSize 3. It starts in
 * GENERATION_MODE so the tests can check the mode is restored. Every call to
 * the model's update(), calculateGradients() and updateWeights() is
 * recorded, with the mode update() was called in.
 */
const setUp = (numberEpochs: number = 1, text: string = TEXT) => {
  const hyperparameters = new Hyperparameters(8, 4, 0.1, Hyperparameters.GENERATION_MODE, 2, 1, 3);
  hyperparameters.numberEpochs = numberEpochs;
  hyperparameters.seed = 42;
  const model = new GPTModel(hyperparameters);

  const calls: { inputs: number[]; mode: number }[] = [];
  const update = model.update.bind(model);
  model.update = (inputs: number[]) => {
    calls.push({ inputs: [...inputs], mode: hyperparameters.mode });
    return update(inputs);
  };

  // The sequence of training calls: "average" or "replace" for calculateGradients(), "update" for updateWeights().
  const trainingCalls: string[] = [];
  const calculateGradients = model.calculateGradients.bind(model);
  model.calculateGradients = (batchAverage: boolean) => {
    trainingCalls.push(batchAverage ? "average" : "replace");
    calculateGradients(batchAverage);
  };
  const updateWeights = model.updateWeights.bind(model);
  model.updateWeights = (optimizer) => {
    trainingCalls.push("update");
    updateWeights(optimizer);
  };

  const trainer = new Trainer(hyperparameters, model);
  trainer.load(new TextDataset(writeTempFile(text)));
  return { hyperparameters, model, trainer, calls, trainingCalls };
};

test("train() trains on every full batch of the training windows and validates on every validation window", () => {
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

test("with windows of different lengths, a batch's gradients are those of the mean loss per token that it reports", () => {
  // No dropout, so the loss only changes when the weights do.
  const hyperparameters = new Hyperparameters(8, 4, 0, Hyperparameters.TRAINING_MODE, 2, 1, 3);
  hyperparameters.seed = 42;
  const model = new GPTModel(hyperparameters);

  // The check compares gradients at one set of weights, so the update at the end of the batch is switched off.
  model.updateWeights = () => {};

  const internals = new Trainer(hyperparameters, model) as unknown as {
    trainOnBatch(batch: { inputs: number[]; targets: number[] }[]): number;
    calculateBatchLoss(batch: { inputs: number[]; targets: number[] }[]): number;
  };
  const batch = [
    { inputs: [464, 3290], targets: [3290, 318] },
    { inputs: [318, 257, 1000, 464], targets: [257, 1000, 464, 3290] },
  ];

  assert.ok(Math.abs(internals.trainOnBatch(batch) - internals.calculateBatchLoss(batch)) <= 1e-9);

  /*
   * checkGradients uses the made-up loss R * (mean loss), but trainOnBatch()
   * always calculates the gradients of the mean loss itself, so the gradients
   * are multiplied by R before they are compared.
   */
  let multiplier = 0;
  const scaled = (gradients: () => ArrayLike<number>) => () => Array.from(gradients(), (gradient) => gradient * multiplier);
  const lastBlock = model.transformers[model.transformers.length - 1];

  const checked = checkGradients({
    forward: () => [[internals.calculateBatchLoss(batch)]],
    backward: (outputErrors) => {
      multiplier = outputErrors[0][0];
      internals.trainOnBatch(batch);
    },
    parameters: [
      // Positions 2 and 3 are only used by the longer window, so a wrong balance between the windows shows up here.
      { name: "position embedding", values: model.embedding.positionEmbeddingMatrix.data, gradients: scaled(() => model.embedding.positionEmbeddingGradients.data) },
      { name: "feed forward secondLayerWeights", values: lastBlock.feedForward.secondLayerWeights.data, gradients: scaled(() => lastBlock.feedForward.secondLayerWeightGradients.data) },
      { name: "output layer", values: model.linearOutputLayer.weights.getRow(3290), gradients: scaled(() => model.linearOutputLayer.weightGradients.getRow(3290)) },
    ],
    // The same step and tolerance as the whole-model check in GPTModel.test.ts, for the same reasons.
    step: 1e-4,
    absoluteTolerance: 1e-9,
  });

  assert.ok(checked > 0);
});

/** Reads the trainer's private windows, to check what load() produced. */
const loadedWindows = (trainer: Trainer) =>
  trainer as unknown as { trainingWindows: Window[]; validationWindows: Window[] };

/** The windows setUp() and the tests below expect tokens to be cut into, with contextLength 4. */
const windowsOf = (tokens: number[]): Window[] => createWindows(tokens, 4);

/** The tokens of the training and validation parts of one text, split 90/10 by characters. */
const split = (text: string) => {
  const tokenizer = new BPETokenizer();
  const splitIndex = Math.floor(0.9 * text.length);
  return { training: tokenizer.encode(text.slice(0, splitIndex)), validation: tokenizer.encode(text.slice(splitIndex)) };
};

test("load() replaces previously loaded windows", () => {
  const { trainer } = setUp();

  trainer.load(new TextDataset(writeTempFile(sentences(10))));

  assert.deepEqual(loadedWindows(trainer).trainingWindows, windowsOf(split(sentences(10)).training));
});

test("load() cuts the windows when it is called, so a later change to contextLength does not affect them", () => {
  const { hyperparameters, trainer } = setUp();
  const before = loadedWindows(trainer).trainingWindows;

  hyperparameters.contextLength = 3;

  assert.equal(loadedWindows(trainer).trainingWindows, before);
  assert.ok(before.every((window) => window.inputs.length === 4));
});

test("load(dataset) trains on the windows the dataset prepares, passing it the settings and a tokenizer", () => {
  const hyperparameters = new Hyperparameters(8, 4, 0.1, Hyperparameters.GENERATION_MODE, 2, 1, 2);
  hyperparameters.seed = 42;
  const model = new GPTModel(hyperparameters);
  const calls: { inputs: number[]; mode: number }[] = [];
  const update = model.update.bind(model);
  model.update = (inputs: number[]) => {
    calls.push({ inputs: [...inputs], mode: hyperparameters.mode });
    return update(inputs);
  };
  const trainer = new Trainer(hyperparameters, model);

  // The windows differ in length, which only a dataset can produce, and which the training loop must accept.
  const training: Window[] = [
    { inputs: [1, 2], targets: [2, 3] },
    { inputs: [4, 5, 6], targets: [5, 6, 7] },
  ];
  const validation: Window[] = [{ inputs: [8], targets: [9] }];
  const received: { hyperparameters: Hyperparameters; tokenizer: BPETokenizer }[] = [];
  const dataset: Dataset = {
    load: (settings, tokenizer) => {
      received.push({ hyperparameters: settings, tokenizer });
      return { training, validation };
    },
  };

  trainer.load(dataset);
  const losses = trainer.train();

  assert.equal(received.length, 1);
  assert.equal(received[0].hyperparameters, hyperparameters);
  assert.ok(received[0].tokenizer instanceof BPETokenizer);
  assert.ok(Number.isFinite(losses[0].trainingLoss) && Number.isFinite(losses[0].validationLoss));
  const inputsIn = (mode: number) => calls.filter((call) => call.mode === mode).map((call) => call.inputs);
  assert.deepEqual(inputsIn(Hyperparameters.TRAINING_MODE).sort((a, b) => a.length - b.length), [[1, 2], [4, 5, 6]]);
  assert.deepEqual(inputsIn(Hyperparameters.VALIDATION_MODE), [[8]]);
});

test("trainingWindowLength lets a short text train on a model with a long context, which would otherwise have no windows", () => {
  const hyperparameters = new Hyperparameters(8, 1024, 0.1, Hyperparameters.GENERATION_MODE, 2, 1, 3);
  hyperparameters.seed = 42;
  const trainer = new Trainer(hyperparameters, new GPTModel(hyperparameters));
  const dataset = new TextDataset(writeTempFile(TEXT));

  trainer.load(dataset);
  assert.throws(() => trainer.train(), /at least batchSize/);

  hyperparameters.trainingWindowLength = 8;
  trainer.load(dataset);
  const losses = trainer.train();

  assert.ok(Number.isFinite(losses[0].trainingLoss) && Number.isFinite(losses[0].validationLoss));
  assert.ok(loadedWindows(trainer).trainingWindows.every((window) => window.inputs.length === 8));
  assert.equal(hyperparameters.contextLength, 1024);
});

test("train() throws a clear error when a dataset gives no validation windows", () => {
  const hyperparameters = new Hyperparameters(8, 4, 0.1, Hyperparameters.TRAINING_MODE, 2, 1, 1);
  const trainer = new Trainer(hyperparameters, new GPTModel(hyperparameters));
  trainer.load({ load: () => ({ training: [{ inputs: [1], targets: [2] }], validation: [] }) });

  assert.throws(() => trainer.train(), /validation set has no windows/);
});

test("train() throws a clear error when there is not enough text for one batch", () => {
  const hyperparameters = new Hyperparameters(8, 4, 0.1, Hyperparameters.TRAINING_MODE, 2, 1, 3);
  const trainer = new Trainer(hyperparameters, new GPTModel(hyperparameters));
  trainer.load(new TextDataset(writeTempFile("Far too short.")));

  assert.throws(() => trainer.train(), /at least batchSize/);
});
