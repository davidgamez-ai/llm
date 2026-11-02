import { test } from "node:test";
import assert from "node:assert/strict";
import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import Hyperparameters from "../Hyperparameters.js";
import GPTModel from "../GPTModel.js";
import Trainer from "../Trainer.js";
import BPETokenizer from "../BPETokenizer.js";
import TinyStoriesDataset from "../TinyStoriesDataset.js";
import { createWindows, joinDocuments } from "../Dataset.js";

/** Writes a CSV file in the TinyStories format, in a temporary folder: a "text" column with one quoted story per record. */
const writeStoriesCsv = (stories: string[]): string => {
  const folder: string = fs.mkdtempSync(path.join(os.tmpdir(), "tinystories-test-"));
  const filePath: string = path.join(folder, "stories.csv");
  fs.writeFileSync(filePath, "text\n" + stories.map((story) => `"${story.replace(/"/g, '""')}"`).join("\n") + "\n", "utf-8");
  return filePath;
};

/** Builds count numbered stories, each containing a quote and a line break as real TinyStories do. */
const makeStories = (count: number, label: string): string[] =>
  Array.from({ length: count }, (_, i) => `${label} story ${i}. Tom said, "Hello ${i}!"\n\nThe end.`);

/** Builds tiny hyperparameters, with contextLength 4 and batchSize 3, for the given numbers of stories. */
const makeHyperparameters = (numberTraining: number, numberValidation: number): Hyperparameters => {
  const hyperparameters = new Hyperparameters(8, 4, 0.1, Hyperparameters.TRAINING_MODE, 2, 1, 3);
  hyperparameters.numberTrainingExamples = numberTraining;
  hyperparameters.numberValidationExamples = numberValidation;
  return hyperparameters;
};

const tokenizer = new BPETokenizer();

/** The windows expected from stories: each encoded separately, then joined with the end-of-text token and cut into windows of 4 tokens. */
const windowsOf = (stories: string[]) =>
  createWindows(joinDocuments(stories.map((story) => tokenizer.encode(story)), tokenizer.endOfTextId), 4);

test("load() uses the first numberTrainingExamples and numberValidationExamples, joined with <|endoftext|> and cut into windows", () => {
  const trainingStories = makeStories(5, "Training");
  const validationStories = makeStories(4, "Validation");
  const dataset = new TinyStoriesDataset(writeStoriesCsv(trainingStories), writeStoriesCsv(validationStories));

  const windows = dataset.load(makeHyperparameters(3, 2), tokenizer);

  assert.deepEqual(windows.training, windowsOf(trainingStories.slice(0, 3)));
  assert.deepEqual(windows.validation, windowsOf(validationStories.slice(0, 2)));
  assert.ok(windows.training.some((window) => window.inputs.includes(tokenizer.endOfTextId)));
});

test("load() cuts windows of trainingWindowLength tokens when it is set", () => {
  const stories = makeStories(2, "Training");
  const hyperparameters = makeHyperparameters(2, 1);
  hyperparameters.trainingWindowLength = 2;

  const { training } = new TinyStoriesDataset(writeStoriesCsv(stories), writeStoriesCsv(makeStories(1, "Validation"))).load(hyperparameters, tokenizer);

  assert.deepEqual(training, createWindows(joinDocuments(stories.map((story) => tokenizer.encode(story)), tokenizer.endOfTextId), 2));
});

test("each load() continues with the next training stories, but always uses the same validation stories", () => {
  const trainingStories = makeStories(6, "Training");
  const validationStories = makeStories(3, "Validation");
  const dataset = new TinyStoriesDataset(writeStoriesCsv(trainingStories), writeStoriesCsv(validationStories));
  const hyperparameters = makeHyperparameters(2, 1);

  assert.equal(dataset.storiesLoaded, 0);
  const first = dataset.load(hyperparameters, tokenizer);
  assert.deepEqual(first.training, windowsOf(trainingStories.slice(0, 2)));
  assert.equal(dataset.storiesLoaded, 2);

  const second = dataset.load(hyperparameters, tokenizer);
  assert.deepEqual(second.training, windowsOf(trainingStories.slice(2, 4)));
  assert.deepEqual(second.validation, first.validation);
  assert.equal(dataset.storiesLoaded, 4);
});

test("a new dataset object starts again from the first story", () => {
  const trainingStories = makeStories(4, "Training");
  const trainingFile = writeStoriesCsv(trainingStories);
  const validationFile = writeStoriesCsv(makeStories(1, "Validation"));
  const hyperparameters = makeHyperparameters(2, 1);
  new TinyStoriesDataset(trainingFile, validationFile).load(hyperparameters, tokenizer);

  const dataset = new TinyStoriesDataset(trainingFile, validationFile);

  assert.deepEqual(dataset.load(hyperparameters, tokenizer).training, windowsOf(trainingStories.slice(0, 2)));
});

test("load() warns when the training file runs out of new stories", (t) => {
  const warnings: string[] = [];
  t.mock.method(console, "warn", (message: string) => warnings.push(message));
  const dataset = new TinyStoriesDataset(writeStoriesCsv(makeStories(3, "Training")), writeStoriesCsv(makeStories(1, "Validation")));
  const hyperparameters = makeHyperparameters(2, 1);

  dataset.load(hyperparameters, tokenizer);
  dataset.load(hyperparameters, tokenizer);

  // Only story 3 was left for the second call.
  assert.equal(dataset.storiesLoaded, 3);
  assert.equal(warnings.length, 1);
  assert.match(warnings[0], /only 1 stories after the first 2; 2 were requested/);
});

test("load() uses every story, with a warning, when a file has fewer than requested", (t) => {
  const warnings: string[] = [];
  t.mock.method(console, "warn", (message: string) => warnings.push(message));
  const trainingStories = makeStories(2, "Training");
  const dataset = new TinyStoriesDataset(writeStoriesCsv(trainingStories), writeStoriesCsv(makeStories(3, "Validation")));

  const windows = dataset.load(makeHyperparameters(10, 1), tokenizer);

  assert.deepEqual(windows.training, windowsOf(trainingStories));
  assert.equal(warnings.length, 1);
  assert.match(warnings[0], /only 2 stories; 10 were requested/);
});

test("a Trainer can load the dataset and train on the stories", () => {
  const hyperparameters = makeHyperparameters(6, 2);
  hyperparameters.mode = Hyperparameters.GENERATION_MODE;
  hyperparameters.seed = 42;
  const trainer = new Trainer(hyperparameters, new GPTModel(hyperparameters));

  trainer.load(new TinyStoriesDataset(writeStoriesCsv(makeStories(6, "Training")), writeStoriesCsv(makeStories(2, "Validation"))));
  const losses = trainer.train();

  assert.equal(losses.length, 1);
  assert.ok(Number.isFinite(losses[0].trainingLoss) && Number.isFinite(losses[0].validationLoss));
});
