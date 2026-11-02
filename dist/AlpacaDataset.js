import * as fs from "fs";
import DEBUG from "./Debug.js";
import createSeededRandom from "./SeededRandom.js";
/**
 * The Alpaca dataset (Taori et al., 2023): instructions paired with responses,
 * for teaching a model to answer requests. Each example becomes one window of
 * the Alpaca prompt followed by the response and the <|endoftext|> token, which
 * is how the model learns to stop answering. Examples differ in length, so the
 * windows do too; they need no padding, because Trainer runs each window
 * through the model separately.
 *
 * The file has no validation data, so it is held out here. The examples are
 * shuffled with SPLIT_SEED, which is separate from Hyperparameters.seed, and the
 * validation examples are taken first. The split is therefore the same in every
 * run, and a validation example is never trained on.
 *
 * Each dataset object keeps its own position in the training examples, so a new
 * object starts again from the beginning.
 */
class AlpacaDataset {
    /**
     * The seed of the shuffle that decides the order of the examples. Changing
     * it changes which examples are held out for validation, so the validation
     * losses of earlier runs could no longer be compared.
     */
    static SPLIT_SEED = 42;
    /** The opening of the prompt for an example that has an input. */
    static PROMPT_WITH_INPUT = "Below is an instruction that describes a task, paired with an input that provides further context. " +
        "Write a response that appropriately completes the request.";
    /** The opening of the prompt for an example that has no input. */
    static PROMPT_WITHOUT_INPUT = "Below is an instruction that describes a task. Write a response that appropriately completes the request.";
    /**
     * Matches an input that only says there is no input. About 600 examples have
     * one of these, in many spellings, such as "No input", "<noinput>" and "N/A",
     * rather than an empty input.
     */
    static PLACEHOLDER_INPUT = /^\W*(no\s*input|none|n\/?a)\W*$/i;
    /** Path to the JSON file. */
    filePath;
    /** The token ids of each example with an output, in shuffled order. Null until the file is first read. */
    examples = null;
    /** The number of examples skipped for having no output. */
    emptyOutputs = 0;
    /** The index in examples at which the next load() continues looking for training examples. */
    nextExample = 0;
    /**
     * Creates a dataset for the given file. Nothing is read until it is needed.
     *
     * @param filePath - Path to the Alpaca JSON file, e.g. data/alpaca_data.json.
     */
    constructor(filePath) {
        this.filePath = filePath;
    }
    /**
     * Builds the prompt for an instruction, ending where the response starts.
     * This is public so that a trained model can be given exactly the template it
     * learned from. An input that is blank or a placeholder such as "No input"
     * uses the template without an input section.
     *
     * @param instruction - The task to carry out.
     * @param input - Further context for the task. May be empty.
     * @returns The prompt.
     */
    static formatPrompt(instruction, input = "") {
        if (input.trim() === "" || AlpacaDataset.PLACEHOLDER_INPUT.test(input)) {
            return `${AlpacaDataset.PROMPT_WITHOUT_INPUT}\n\n### Instruction:\n${instruction}\n\n### Response:\n`;
        }
        return `${AlpacaDataset.PROMPT_WITH_INPUT}\n\n### Instruction:\n${instruction}\n\n### Input:\n${input}\n\n### Response:\n`;
    }
    /**
     * Prepares windows, one per example.
     *
     * Validation always uses the first numberValidationExamples that fit in a
     * window, so that the validation losses of different runs can be compared.
     * Each call trains on the next numberTrainingExamples that fit after them,
     * continuing from where the previous call finished, so every training run
     * sees new examples.
     *
     * Examples too long for trainingWindowLength are dropped rather than
     * truncated: truncating would remove the end of the response and the end
     * token, teaching the model to stop mid-sentence. If fewer examples are left
     * than requested, all of them are used and a warning is printed.
     *
     * @param hyperparameters - The source of numberTrainingExamples, numberValidationExamples and trainingWindowLength.
     * @param tokenizer - Converts the examples into token ids.
     * @returns The training and validation windows.
     * @throws Error if the file cannot be read or is not in the Alpaca format.
     */
    load(hyperparameters, tokenizer) {
        const examples = this.prepareExamples(tokenizer);
        const windowLength = hyperparameters.trainingWindowLength;
        // The search for validation examples always starts from the beginning, so it always finds the same ones.
        const validation = [];
        let index = 0;
        for (; index < examples.length && validation.length < hyperparameters.numberValidationExamples; index++) {
            if (AlpacaDataset.fits(examples[index], windowLength)) {
                validation.push(AlpacaDataset.createWindow(examples[index]));
            }
        }
        // Training continues after the validation examples, and after the examples earlier calls used.
        const training = [];
        for (index = Math.max(index, this.nextExample); index < examples.length && training.length < hyperparameters.numberTrainingExamples; index++) {
            if (AlpacaDataset.fits(examples[index], windowLength)) {
                training.push(AlpacaDataset.createWindow(examples[index]));
            }
        }
        this.nextExample = index;
        this.warnIfShort("training", training.length, hyperparameters.numberTrainingExamples);
        this.warnIfShort("validation", validation.length, hyperparameters.numberValidationExamples);
        if (DEBUG.TRAINER) {
            const counts = this.countExamples(hyperparameters, tokenizer);
            console.log(`AlpacaDataset| Examples up to ${windowLength} tokens long: ${counts.kept} kept, ${counts.tooLong} dropped as too long, ` +
                `${counts.emptyOutput} skipped as empty. Training examples: ${training.length}. Validation examples: ${validation.length}.`);
        }
        return { training, validation };
    }
    /**
     * Counts how the examples in the file are used at the current window length.
     *
     * @param hyperparameters - The source of trainingWindowLength.
     * @param tokenizer - Converts the examples into token ids.
     * @returns The numbers of examples kept, dropped as too long and skipped as empty.
     * @throws Error if the file cannot be read or is not in the Alpaca format.
     */
    countExamples(hyperparameters, tokenizer) {
        const examples = this.prepareExamples(tokenizer);
        const kept = examples.filter((example) => AlpacaDataset.fits(example, hyperparameters.trainingWindowLength)).length;
        return { kept, tooLong: examples.length - kept, emptyOutput: this.emptyOutputs };
    }
    /**
     * Reads, shuffles and encodes the examples the first time they are needed.
     * All of them are encoded, however few are used, because a window can only be
     * chosen once its length is known. This takes only a few seconds.
     *
     * @param tokenizer - Converts the examples into token ids.
     * @returns The token ids of each example with an output, in shuffled order, each ending with the end token.
     * @throws Error if the file cannot be read or is not in the Alpaca format.
     */
    prepareExamples(tokenizer) {
        if (this.examples !== null) {
            return this.examples;
        }
        const records = AlpacaDataset.readRecords(this.filePath);
        const withOutputs = records.filter((record) => record.output.trim() !== "");
        this.emptyOutputs = records.length - withOutputs.length;
        this.examples = AlpacaDataset.shuffle(withOutputs).map((record) => {
            const tokens = tokenizer.encode(AlpacaDataset.formatPrompt(record.instruction, record.input) + record.output);
            tokens.push(tokenizer.endOfTextId);
            return tokens;
        });
        return this.examples;
    }
    /**
     * Reads the examples from a JSON file and checks they have the expected form.
     *
     * @param filePath - Path to the JSON file.
     * @returns The examples, in file order.
     * @throws Error if the file cannot be read or is not in the Alpaca format.
     */
    static readRecords(filePath) {
        const records = JSON.parse(fs.readFileSync(filePath, "utf-8"));
        const isExample = (record) => typeof record?.instruction === "string" && typeof record?.input === "string" && typeof record?.output === "string";
        if (!Array.isArray(records) || !records.every(isExample)) {
            throw new Error(`AlpacaDataset| ${filePath} must hold an array of objects with instruction, input and output text`);
        }
        return records;
    }
    /**
     * Returns a shuffled copy of the examples, using the Fisher-Yates shuffle
     * and SPLIT_SEED, so the order is the same in every run.
     *
     * @param records - The examples to shuffle. Not changed.
     * @returns A new array holding the same examples in shuffled order.
     */
    static shuffle(records) {
        const random = createSeededRandom(AlpacaDataset.SPLIT_SEED);
        const shuffled = [...records];
        // Work backwards, swapping each example with a randomly chosen one at or before it.
        for (let i = shuffled.length - 1; i > 0; --i) {
            const j = Math.floor(random() * (i + 1));
            [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
        }
        return shuffled;
    }
    /**
     * Checks whether an example fits in one window. An example of n tokens gives
     * a window of n - 1 inputs, since the first token is never a target and the
     * last is never an input.
     *
     * @param tokens - The token ids of the example.
     * @param windowLength - The longest window allowed.
     * @returns True if the example fits.
     */
    static fits(tokens, windowLength) {
        return tokens.length - 1 <= windowLength;
    }
    /**
     * Makes the window for an example: every token but the last as the inputs,
     * and the same tokens shifted one place to the right as the targets.
     *
     * @param tokens - The token ids of the example.
     * @returns The window.
     */
    static createWindow(tokens) {
        return { inputs: tokens.slice(0, -1), targets: tokens.slice(1) };
    }
    /**
     * Warns if fewer examples were found than were requested.
     *
     * @param description - Which set of examples this is, shown in the warning.
     * @param found - The number of examples found.
     * @param requested - The number of examples requested.
     */
    warnIfShort(description, found, requested) {
        if (found < requested) {
            console.warn(`AlpacaDataset| ${this.filePath} has only ${found} ${description} examples left that fit; ${requested} were requested.`);
        }
    }
}
export default AlpacaDataset;
