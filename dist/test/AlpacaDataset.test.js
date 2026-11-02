import { test } from "node:test";
import assert from "node:assert/strict";
import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import Hyperparameters from "../Hyperparameters.js";
import GPTModel from "../GPTModel.js";
import Trainer from "../Trainer.js";
import BPETokenizer from "../BPETokenizer.js";
import AlpacaDataset from "../AlpacaDataset.js";
/** Writes examples to a JSON file in a temporary folder and returns its path. */
const writeExamples = (examples) => {
    const folder = fs.mkdtempSync(path.join(os.tmpdir(), "alpaca-test-"));
    const filePath = path.join(folder, "alpaca.json");
    fs.writeFileSync(filePath, JSON.stringify(examples), "utf-8");
    return filePath;
};
/** Builds count numbered examples with no input. */
const makeExamples = (count) => Array.from({ length: count }, (_, i) => ({ instruction: `Name item ${i}.`, input: "", output: `It is item ${i}.` }));
/** Builds hyperparameters for a tiny model with a context long enough for a whole example, and the given numbers of examples. */
const makeHyperparameters = (numberTraining, numberValidation) => {
    const hyperparameters = new Hyperparameters(8, 128, 0.1, Hyperparameters.TRAINING_MODE, 2, 1, 2);
    hyperparameters.numberTrainingExamples = numberTraining;
    hyperparameters.numberValidationExamples = numberValidation;
    return hyperparameters;
};
const tokenizer = new BPETokenizer();
/** The text of the example a window was made from, including its last token. */
const textOf = (window) => tokenizer.decode([...window.inputs, window.targets[window.targets.length - 1]]);
test("formatPrompt() gives the exact Alpaca template with an input", () => {
    assert.equal(AlpacaDataset.formatPrompt("Add the numbers.", "2 and 3"), "Below is an instruction that describes a task, paired with an input that provides further context. " +
        "Write a response that appropriately completes the request.\n\n" +
        "### Instruction:\nAdd the numbers.\n\n### Input:\n2 and 3\n\n### Response:\n");
});
test("formatPrompt() gives the exact Alpaca template without an input", () => {
    const expected = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n" +
        "### Instruction:\nName a colour.\n\n### Response:\n";
    assert.equal(AlpacaDataset.formatPrompt("Name a colour.", ""), expected);
    assert.equal(AlpacaDataset.formatPrompt("Name a colour."), expected);
    assert.equal(AlpacaDataset.formatPrompt("Name a colour.", "   "), expected);
});
test("formatPrompt() treats placeholder inputs as no input, but keeps real ones", () => {
    const noInput = AlpacaDataset.formatPrompt("Name a colour.");
    for (const placeholder of ["No input", "no input.", "Noinput", "<noinput>", "<No Input>", "\"<noinput>\"", "NoInput", "N/A", "None"]) {
        assert.equal(AlpacaDataset.formatPrompt("Name a colour.", placeholder), noInput, placeholder);
    }
    // Text that only starts like a placeholder, and text with no Latin letters, is real input.
    for (const input of ["No input was given by the user", "Nonetheless", "你好"]) {
        assert.match(AlpacaDataset.formatPrompt("Name a colour.", input), /### Input:/, input);
    }
});
test("each example becomes one window: the prompt, the output and the end token, with targets shifted by one", (t) => {
    // A file of one example has none left to train on, which load() warns about.
    t.mock.method(console, "warn", () => { });
    const example = { instruction: "Add the numbers.", input: "2 and 3", output: "The answer is 5." };
    const { training, validation } = new AlpacaDataset(writeExamples([example])).load(makeHyperparameters(1, 1), tokenizer);
    assert.equal(training.length, 0);
    assert.equal(validation.length, 1);
    const window = validation[0];
    assert.equal(textOf(window), `${AlpacaDataset.formatPrompt(example.instruction, example.input)}${example.output}${BPETokenizer.END_OF_TEXT}`);
    assert.deepEqual(window.targets.slice(0, -1), window.inputs.slice(1));
    assert.equal(window.targets.length, window.inputs.length);
    assert.equal(window.targets[window.targets.length - 1], tokenizer.endOfTextId);
    assert.ok(!window.inputs.includes(tokenizer.endOfTextId));
});
test("examples that are too long or have empty outputs are left out and counted", () => {
    const short = makeExamples(4);
    const long = { instruction: "Tell a long story.", input: "", output: "and then ".repeat(100) };
    const empty = [{ instruction: "Say nothing.", input: "", output: "" }, { instruction: "Say nothing again.", input: "", output: "  \n" }];
    const dataset = new AlpacaDataset(writeExamples([...short, long, ...empty]));
    const hyperparameters = makeHyperparameters(3, 1);
    const windows = dataset.load(hyperparameters, tokenizer);
    assert.deepEqual(dataset.countExamples(hyperparameters, tokenizer), { kept: 4, tooLong: 1, emptyOutput: 2 });
    const texts = [...windows.training, ...windows.validation].map(textOf);
    assert.equal(texts.length, 4);
    assert.ok(texts.every((text) => /It is item \d\./.test(text)));
});
test("an example exactly as long as the window is kept, and one token longer is dropped", () => {
    const example = makeExamples(1)[0];
    const tokens = tokenizer.encode(AlpacaDataset.formatPrompt(example.instruction) + example.output).length + 1;
    const hyperparameters = makeHyperparameters(1, 1);
    const dataset = new AlpacaDataset(writeExamples([example]));
    // n tokens give a window of n - 1 inputs.
    hyperparameters.trainingWindowLength = tokens - 1;
    assert.equal(dataset.countExamples(hyperparameters, tokenizer).kept, 1);
    hyperparameters.trainingWindowLength = tokens - 2;
    assert.equal(dataset.countExamples(hyperparameters, tokenizer).kept, 0);
});
test("no validation example appears in the training windows", () => {
    const windows = new AlpacaDataset(writeExamples(makeExamples(20))).load(makeHyperparameters(15, 5), tokenizer);
    const validation = new Set(windows.validation.map(textOf));
    assert.equal(validation.size, 5);
    assert.equal(windows.training.length, 15);
    assert.ok(windows.training.every((window) => !validation.has(textOf(window))));
});
test("the examples are shuffled, and the split is the same for any Hyperparameters.seed", () => {
    const filePath = writeExamples(makeExamples(20));
    const first = makeHyperparameters(10, 5);
    first.seed = 1;
    const second = makeHyperparameters(10, 5);
    second.seed = 2;
    const windows = new AlpacaDataset(filePath).load(first, tokenizer);
    assert.deepEqual(new AlpacaDataset(filePath).load(second, tokenizer), windows);
    // Hyperparameters' stream is not used, so a run with a seed trains the same as it would otherwise.
    assert.equal(first.random(), makeSeededFirstValue(1));
    // In file order the validation examples would be items 0 to 4.
    assert.notDeepEqual(windows.validation.map(textOf), makeExamples(5).map((example) => `${AlpacaDataset.formatPrompt(example.instruction)}${example.output}${BPETokenizer.END_OF_TEXT}`));
});
/** The first value of a new stream with the given seed, to check that nothing has drawn from a stream. */
function makeSeededFirstValue(seed) {
    const hyperparameters = new Hyperparameters();
    hyperparameters.seed = seed;
    return hyperparameters.random();
}
test("a second load() gives new training examples and the same validation examples", () => {
    const dataset = new AlpacaDataset(writeExamples(makeExamples(20)));
    const hyperparameters = makeHyperparameters(5, 3);
    const first = dataset.load(hyperparameters, tokenizer);
    const second = dataset.load(hyperparameters, tokenizer);
    assert.deepEqual(second.validation, first.validation);
    const firstTraining = new Set(first.training.map(textOf));
    assert.equal(second.training.length, 5);
    assert.ok(second.training.every((window) => !firstTraining.has(textOf(window))));
});
test("a new dataset object starts again from the beginning", () => {
    const filePath = writeExamples(makeExamples(20));
    const hyperparameters = makeHyperparameters(5, 3);
    const first = new AlpacaDataset(filePath).load(hyperparameters, tokenizer);
    assert.deepEqual(new AlpacaDataset(filePath).load(hyperparameters, tokenizer), first);
});
test("load() uses what remains, with a warning, when too few examples are left", (t) => {
    const warnings = [];
    t.mock.method(console, "warn", (message) => warnings.push(message));
    const dataset = new AlpacaDataset(writeExamples(makeExamples(7)));
    const hyperparameters = makeHyperparameters(4, 2);
    dataset.load(hyperparameters, tokenizer);
    assert.equal(warnings.length, 0);
    // 7 examples, less 2 for validation and 4 already used, leaves 1.
    assert.equal(dataset.load(hyperparameters, tokenizer).training.length, 1);
    assert.equal(warnings.length, 1);
    assert.match(warnings[0], /only 1 training examples left that fit; 4 were requested/);
});
test("load() rejects a file that is not in the Alpaca format", () => {
    const hyperparameters = makeHyperparameters(1, 1);
    for (const contents of [{ instruction: "Not an array." }, [{ instruction: "No output.", input: "" }], ["text"], [null]]) {
        assert.throws(() => new AlpacaDataset(writeExamples(contents)).load(hyperparameters, tokenizer), /instruction, input and output/);
    }
});
test("a Trainer can load the dataset and train on windows of different lengths", () => {
    const hyperparameters = makeHyperparameters(4, 2);
    hyperparameters.mode = Hyperparameters.GENERATION_MODE;
    hyperparameters.seed = 42;
    // Every example is a different length, wherever the shuffle puts it.
    const examples = makeExamples(6).map((example, i) => ({ ...example, output: "more ".repeat(i + 1).trim() }));
    const filePath = writeExamples(examples);
    const trainer = new Trainer(hyperparameters, new GPTModel(hyperparameters));
    trainer.load(new AlpacaDataset(filePath));
    const losses = trainer.train();
    const { training } = new AlpacaDataset(filePath).load(hyperparameters, tokenizer);
    assert.equal(new Set(training.map((window) => window.inputs.length)).size, 4);
    assert.ok(Number.isFinite(losses[0].trainingLoss) && Number.isFinite(losses[0].validationLoss));
});
