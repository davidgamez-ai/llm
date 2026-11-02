import { test } from "node:test";
import assert from "node:assert/strict";
import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import Hyperparameters from "../Hyperparameters.js";
import BPETokenizer from "../BPETokenizer.js";
import TextDataset from "../TextDataset.js";
import { createWindows } from "../Dataset.js";

/** Writes text to a new file in a temporary folder and returns its path. */
const writeTempFile = (text: string): string => {
  const folder: string = fs.mkdtempSync(path.join(os.tmpdir(), "text-dataset-test-"));
  const filePath: string = path.join(folder, "data.txt");
  fs.writeFileSync(filePath, text, "utf-8");
  return filePath;
};

/** Builds varied text from count numbered sentences. */
const sentences = (count: number): string =>
  Array.from({ length: count }, (_, i) => `Sentence ${i} is about item ${(i * 7) % 13}.`).join(" ");

const tokenizer = new BPETokenizer();

/** Tiny hyperparameters, with contextLength 4. */
const makeHyperparameters = (): Hyperparameters => new Hyperparameters(8, 4, 0.1, Hyperparameters.TRAINING_MODE, 2, 1, 3);

/** The windows of 4 tokens expected from token ids. */
const windowsOf = (tokens: number[]) => createWindows(tokens, 4);

/** The tokens of the training and validation parts of one text, split at the given ratio by characters. */
const split = (text: string, ratio: number = 0.9) => {
  const splitIndex = Math.floor(ratio * text.length);
  return { training: tokenizer.encode(text.slice(0, splitIndex)), validation: tokenizer.encode(text.slice(splitIndex)) };
};

test("load() splits a file 90/10 by characters before tokenizing, and adds no end-of-text token", () => {
  const text = sentences(40);

  const windows = new TextDataset(writeTempFile(text)).load(makeHyperparameters(), tokenizer);

  assert.deepEqual(windows.training, windowsOf(split(text).training));
  assert.deepEqual(windows.validation, windowsOf(split(text).validation));
  assert.ok(windows.training.every((window) => !window.inputs.includes(tokenizer.endOfTextId)));
});

test("load() uses trainingRatio to split the text", () => {
  const text = sentences(40);
  const hyperparameters = makeHyperparameters();
  hyperparameters.trainingRatio = 0.5;

  const windows = new TextDataset(writeTempFile(text)).load(hyperparameters, tokenizer);

  assert.deepEqual(windows.training, windowsOf(split(text, 0.5).training));
  assert.deepEqual(windows.validation, windowsOf(split(text, 0.5).validation));
});

test("each window's targets are its inputs shifted one place to the right", () => {
  const { training } = new TextDataset(writeTempFile(sentences(40))).load(makeHyperparameters(), tokenizer);

  assert.ok(training.length > 0);
  for (const window of training) {
    assert.equal(window.inputs.length, 4);
    assert.deepEqual(window.targets.slice(0, -1), window.inputs.slice(1));
  }
});

test("load() with several files splits each file 90/10 and joins the parts with <|endoftext|>", () => {
  const first = sentences(30);
  const second = "A different book about other things, which goes on for a while. ".repeat(20);

  const windows = new TextDataset([writeTempFile(first), writeTempFile(second)]).load(makeHyperparameters(), tokenizer);

  // Each set holds the matching part of every file, in order, separated by the end-of-text token.
  const a = split(first);
  const b = split(second);
  assert.deepEqual(windows.training, windowsOf([...a.training, tokenizer.endOfTextId, ...b.training]));
  assert.deepEqual(windows.validation, windowsOf([...a.validation, tokenizer.endOfTextId, ...b.validation]));
});

test("one path gives the same windows as an array holding that path", () => {
  const filePath = writeTempFile(sentences(40));
  const hyperparameters = makeHyperparameters();

  assert.deepEqual(new TextDataset(filePath).load(hyperparameters, tokenizer), new TextDataset([filePath]).load(hyperparameters, tokenizer));
});

test("every load() gives the same windows", () => {
  const dataset = new TextDataset(writeTempFile(sentences(40)));
  const hyperparameters = makeHyperparameters();

  assert.deepEqual(dataset.load(hyperparameters, tokenizer), dataset.load(hyperparameters, tokenizer));
});

test("maximumCharacters takes characters from the start of each file before it is split", () => {
  const text = sentences(40);
  const maximumCharacters = 300;

  const windows = new TextDataset(writeTempFile(text), { maximumCharacters }).load(makeHyperparameters(), tokenizer);

  // The validation part comes from within the characters used, not from the end of the file.
  assert.deepEqual(windows.training, windowsOf(split(text.slice(0, maximumCharacters)).training));
  assert.deepEqual(windows.validation, windowsOf(split(text.slice(0, maximumCharacters)).validation));
});

test("maximumCharacters applies to each file separately", () => {
  const first = sentences(30);
  const second = "A different book about other things, which goes on for a while. ".repeat(20);
  const maximumCharacters = 300;

  const windows = new TextDataset([writeTempFile(first), writeTempFile(second)], { maximumCharacters }).load(makeHyperparameters(), tokenizer);

  const a = split(first.slice(0, maximumCharacters));
  const b = split(second.slice(0, maximumCharacters));
  assert.deepEqual(windows.training, windowsOf([...a.training, tokenizer.endOfTextId, ...b.training]));
});

test("maximumCharacters larger than the file uses the whole file", () => {
  const text = sentences(40);
  const hyperparameters = makeHyperparameters();
  const filePath = writeTempFile(text);

  assert.deepEqual(
    new TextDataset(filePath, { maximumCharacters: text.length * 2 }).load(hyperparameters, tokenizer),
    new TextDataset(filePath).load(hyperparameters, tokenizer)
  );
});

test("load() cuts windows of trainingWindowLength tokens when it is set", () => {
  const text = sentences(40);
  const hyperparameters = makeHyperparameters();
  hyperparameters.trainingWindowLength = 2;

  const windows = new TextDataset(writeTempFile(text)).load(hyperparameters, tokenizer);

  assert.deepEqual(windows.training, createWindows(split(text).training, 2));
  assert.deepEqual(windows.validation, createWindows(split(text).validation, 2));
});

test("the constructor rejects an empty array of files", () => {
  assert.throws(() => new TextDataset([]), /at least one file/);
});

test("the constructor rejects a maximumCharacters that is not a positive whole number", () => {
  for (const maximumCharacters of [0, -1, 1.5, NaN, Infinity]) {
    assert.throws(() => new TextDataset("unused.txt", { maximumCharacters }), RangeError, `accepted ${maximumCharacters}`);
  }
});
