import { test } from "node:test";
import assert from "node:assert/strict";
import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import Hyperparameters from "../Hyperparameters.js";
import GPTModel from "../GPTModel.js";
import ModelManager from "../ModelManager.js";
import BPETokenizer from "../BPETokenizer.js";
import { crossEntropyLoss } from "../Loss.js";

/*
 * Returns the path of a new file in a temporary folder, so the tests do not
 * write into the project.
 */
const temporaryFile = (): string => {
  const folder: string = fs.mkdtempSync(path.join(os.tmpdir(), "model-manager-test-"));
  return path.join(folder, "model.safetensors");
};

/*
 * Creates a tiny model: embeddingSize 8, contextLength 4, no dropout, 2
 * heads and 2 transformer blocks. Different seeds give different weights.
 */
const createModel = (seed: number, numberTransformerBlocks: number = 2): GPTModel => {
  const hyperparameters = new Hyperparameters(8, 4, 0, Hyperparameters.VALIDATION_MODE, 2, numberTransformerBlocks);
  hyperparameters.seed = seed;
  return new GPTModel(hyperparameters);
};

/** Copies every parameter array in a model, so its values can be compared after it has changed. */
const copyParameters = (model: GPTModel): Float32Array[] => model.getParameters().map(({ values }) => values.slice());

test("getParameters() lists every parameter exactly once, with matching shapes", () => {
  const model = createModel(1);
  const parameters = model.getParameters();

  assert.equal(new Set(parameters.map(({ name }) => name)).size, parameters.length, "names are unique");
  assert.equal(parameters.reduce((sum, { values }) => sum + values.length, 0), model.getParameterCount());
  for (const { name, shape, values } of parameters) {
    assert.equal(shape.reduce((product, size) => product * size, 1), values.length, name);
  }
});

test("load() restores exactly the weights that save() wrote", () => {
  const fileName = temporaryFile();
  const saved = createModel(1);
  new ModelManager().save(saved, fileName);

  const loaded = createModel(2);
  assert.notDeepEqual(copyParameters(loaded), copyParameters(saved), "the models start with different weights");

  new ModelManager().load(loaded, fileName);
  assert.deepEqual(copyParameters(loaded), copyParameters(saved));

  // The loaded model gives the same logits as the saved one.
  const tokens = [464, 3290, 318];
  assert.deepEqual(loaded.update(tokens), saved.update(tokens));
});

test("load() copies into the model's existing arrays rather than replacing them", () => {
  const fileName = temporaryFile();
  new ModelManager().save(createModel(1), fileName);

  const model = createModel(2);
  const arraysBefore = model.getParameters().map(({ values }) => values);
  new ModelManager().load(model, fileName);
  model.getParameters().forEach(({ name, values }, i) => assert.equal(values, arraysBefore[i], name));
});

test("save() writes a valid safetensors file: little-endian header length, padded JSON header, 4 bytes per parameter", () => {
  const fileName = temporaryFile();
  const model = createModel(1);
  new ModelManager().save(model, fileName);

  const bytes = fs.readFileSync(fileName);
  const headerLength = Number(bytes.readBigUInt64LE(0));
  assert.equal(headerLength % 8, 0, "the header is padded to a multiple of 8 bytes");
  assert.equal(bytes.length, 8 + headerLength + 4 * model.getParameterCount());

  const header = JSON.parse(bytes.subarray(8, 8 + headerLength).toString("utf-8"));
  const first = model.getParameters()[0];
  assert.deepEqual(header[first.name], { dtype: "F32", shape: first.shape, data_offsets: [0, first.values.byteLength] });

  // The first value in the data section is the first value of the first array.
  assert.equal(bytes.readFloatLE(8 + headerLength), first.values[0]);
});

test("load() rejects a file saved from a model with different hyperparameters, leaving the model unchanged", () => {
  const fileName = temporaryFile();
  new ModelManager().save(createModel(1, 3), fileName);

  // Fewer blocks: the file has arrays the model does not.
  const model = createModel(2);
  const before = copyParameters(model);
  assert.throws(() => new ModelManager().load(model, fileName), RangeError);
  assert.deepEqual(copyParameters(model), before);

  // More blocks: the model has arrays the file does not.
  assert.throws(() => new ModelManager().load(createModel(2, 4), fileName), RangeError);
});

test("load() names the hyperparameter that differs when the model does not match the file", () => {
  const fileName = temporaryFile();
  new ModelManager().save(createModel(1, 3), fileName);

  assert.throws(() => new ModelManager().load(createModel(2), fileName), /numberTransformerBlocks 3; this model has 2/);
});

test("load() accepts a model whose training hyperparameters differ from the file's", () => {
  const fileName = temporaryFile();
  const saved = createModel(1);
  new ModelManager().save(saved, fileName);

  const model = createModel(2);
  model.hyperparameters.learningRate = 0.123;
  new ModelManager().load(model, fileName);
  assert.deepEqual(copyParameters(model), copyParameters(saved));
  assert.equal(model.hyperparameters.learningRate, 0.123, "the model keeps its own training hyperparameters");
});

test("Hyperparameters.fromJSON() rebuilds the values written by toJSON(), with or without a seed", () => {
  const hyperparameters = Hyperparameters.forReducedTraining(3);
  hyperparameters.qkvBias = false;
  assert.deepEqual(Hyperparameters.fromJSON(JSON.parse(JSON.stringify(hyperparameters))).toJSON(), hyperparameters.toJSON());

  hyperparameters.seed = 99;
  const rebuilt = Hyperparameters.fromJSON(JSON.parse(JSON.stringify(hyperparameters)));
  assert.equal(rebuilt.seed, 99);
  assert.equal(rebuilt.weightMatrixColumns, hyperparameters.weightMatrixColumns);
});

test("Hyperparameters.fromJSON() validates the values", () => {
  assert.throws(() => Hyperparameters.fromJSON({ embeddingSize: 10, numberAttentionHeads: 3 }), RangeError);
  assert.throws(() => Hyperparameters.fromJSON({ qkvBias: "yes" as unknown as boolean }), TypeError);
});

test("loadModel() builds a model with the saved hyperparameters and weights, in generation mode by default", () => {
  const fileName = temporaryFile();
  const saved = createModel(1, 3);
  saved.hyperparameters.learningRate = 0.005;
  new ModelManager().save(saved, fileName);

  const loaded = new ModelManager().loadModel(fileName);
  assert.equal(loaded.hyperparameters.mode, Hyperparameters.GENERATION_MODE);
  assert.deepEqual({ ...loaded.hyperparameters.toJSON(), mode: saved.hyperparameters.mode }, saved.hyperparameters.toJSON());
  assert.deepEqual(copyParameters(loaded), copyParameters(saved));

  assert.equal(new ModelManager().loadModel(fileName, Hyperparameters.TRAINING_MODE).hyperparameters.mode, Hyperparameters.TRAINING_MODE);
});

test("loadModel() explains that a file without saved hyperparameters needs load()", () => {
  const fileName = temporaryFile();
  new ModelManager().save(createModel(1), fileName);

  // Rewrite the file with the hyperparameters removed from its metadata, as another program might write it.
  const bytes = fs.readFileSync(fileName);
  const headerLength = Number(bytes.readBigUInt64LE(0));
  const header = JSON.parse(bytes.subarray(8, 8 + headerLength).toString("utf-8"));
  delete header.__metadata__.hyperparameters;
  const newHeader = Buffer.from(JSON.stringify(header), "utf-8");
  const newLength = Buffer.alloc(8);
  newLength.writeBigUInt64LE(BigInt(newHeader.length));
  fs.writeFileSync(fileName, Buffer.concat([newLength, newHeader, bytes.subarray(8 + headerLength)]));

  assert.throws(() => new ModelManager().loadModel(fileName), /no saved hyperparameters/);

  // load() still works, relying on the shape checks alone.
  const saved = createModel(1);
  const model = createModel(2);
  new ModelManager().load(model, fileName);
  assert.deepEqual(copyParameters(model), copyParameters(saved));
});

/*
 * Loading OpenAI's GPT-2 small weights. The 548 MB file is not part of the
 * repository, so these tests are skipped when it has not been downloaded.
 */

const GPT2_MISSING: string | false = !fs.existsSync(ModelManager.GPT2_SMALL_FILE) && "models/gpt2-small.safetensors has not been downloaded";

test("the default GPT-2 small file is gpt2-small.safetensors in the project's models folder", () => {
  assert.equal(path.basename(ModelManager.GPT2_SMALL_FILE), "gpt2-small.safetensors");
  assert.equal(path.basename(path.dirname(ModelManager.GPT2_SMALL_FILE)), "models");
  // The models folder is beside dist/, at the root of the project.
  assert.ok(fs.existsSync(path.join(path.dirname(ModelManager.GPT2_SMALL_FILE), "..", "package.json")));
});

test("the GPT-2 small weights are present in the models folder", { skip: GPT2_MISSING }, () => {
  assert.equal(fs.statSync(ModelManager.GPT2_SMALL_FILE).size, 548105171);
});

test("loadGPT2() explains how to download the weights when the file does not exist", () => {
  const fileName = path.join(os.tmpdir(), "no-such-folder", "gpt2-small.safetensors");
  assert.throws(() => new ModelManager().loadGPT2(fileName), /were not found at .*Download them from https:\/\/huggingface\.co/);
});

test("loadGPT2() rejects a safetensors file that is not GPT-2 small, before building the model", () => {
  const fileName = temporaryFile();
  new ModelManager().save(createModel(1), fileName);
  assert.throws(() => new ModelManager().loadGPT2(fileName), /has no array called wte\.weight/);
});

test("loadGPT2() loads GPT-2 small, which predicts English text far better than chance", { skip: GPT2_MISSING }, () => {
  let lastProgress = 0;
  const model = new ModelManager().loadGPT2(undefined, undefined, (done, total) => (lastProgress = done / total));
  assert.equal(lastProgress, 1);
  assert.equal(model.hyperparameters.mode, Hyperparameters.GENERATION_MODE);

  // The output layer is a copy of the token embedding, as GPT-2 ties them.
  assert.deepEqual(model.linearOutputLayer.weights.data, model.embedding.textEmbeddingMatrix.data);

  /*
   * The average loss of predicting each next token of an ordinary sentence.
   * Guessing at random gives ln(50,257) = 10.8; GPT-2 small gives about 3
   * to 4, and any mistake in mapping the weights (e.g. heads or Q/K/V in the
   * wrong order) makes it far worse.
   */
  model.hyperparameters.mode = Hyperparameters.VALIDATION_MODE;
  const tokens: number[] = new BPETokenizer().encode("The quick brown fox jumps over the lazy dog. It was a sunny day, and the children were playing in the park.");
  const logits = model.update(tokens.slice(0, -1));
  const loss = logits.reduce((sum, row, i) => sum + crossEntropyLoss(row, tokens[i + 1]), 0) / logits.length;
  assert.ok(loss < 4.5, `average loss ${loss.toFixed(3)}`);

  /*
   * GPT-2 small's three most likely next tokens after "The capital of
   * France is the city of" are French cities: " Marse" (the start of
   * Marseille), " Paris" and " Lyon", in that order.
   */
  model.hyperparameters.mode = Hyperparameters.GENERATION_MODE;
  const tokenizer = new BPETokenizer();
  const next = model.update(tokenizer.encode("The capital of France is the city of"))[0];
  const topThree = [...next.keys()].sort((a, b) => next[b] - next[a]).slice(0, 3);
  assert.deepEqual(topThree.map((id) => tokenizer.decode([id])), [" Marse", " Paris", " Lyon"]);
});

test("load() rejects a truncated file", () => {
  const fileName = temporaryFile();
  new ModelManager().save(createModel(1), fileName);
  const bytes = fs.readFileSync(fileName);
  fs.writeFileSync(fileName, bytes.subarray(0, bytes.length - 4));

  assert.throws(() => new ModelManager().load(createModel(2), fileName), RangeError);
});
