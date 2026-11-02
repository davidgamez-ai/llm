import * as fs from "fs";
import { createWindows, joinDocuments } from "./Dataset.js";
import DEBUG from "./Debug.js";
/**
 * One or more continuous text files, such as The Verdict or Tiny Shakespeare.
 * Every call of load() gives the same windows, so a text is repeated by
 * training for more epochs rather than by loading it again.
 */
class TextDataset {
    /** Paths to the text files, in the order they are joined. */
    filePaths;
    /** The number of characters to use from the start of each file, or undefined to use all of them. */
    maximumCharacters;
    /**
     * Creates a dataset for the given files. Nothing is read until load() is
     * called.
     *
     * @param filePaths - Path to a UTF-8 text file, or an array of paths.
     * @param options - Optional settings; see TextDatasetOptions.
     * @throws Error if filePaths is an empty array.
     * @throws RangeError if maximumCharacters is not a positive whole number.
     */
    constructor(filePaths, options = {}) {
        this.filePaths = typeof filePaths === "string" ? [filePaths] : filePaths;
        if (this.filePaths.length === 0) {
            throw new Error("TextDataset| A dataset needs at least one file");
        }
        const maximumCharacters = options.maximumCharacters;
        if (maximumCharacters !== undefined && !(Number.isInteger(maximumCharacters) && maximumCharacters > 0)) {
            throw new RangeError(`TextDataset| maximumCharacters must be a positive whole number; got ${maximumCharacters}`);
        }
        this.maximumCharacters = maximumCharacters;
    }
    /**
     * Reads the files, splits each into a training part and a validation part,
     * encodes the parts with the tokenizer, and cuts them into windows of
     * trainingWindowLength tokens.
     *
     * As in the book, each file's text is split by characters before it is
     * tokenized: its first trainingRatio of the characters are used for
     * training and the rest for validation. Each file is split separately,
     * rather than joining the files and splitting once, so the validation set
     * contains a sample of every file, not just the end of the last one.
     *
     * The training parts of the files are then joined, in the order given,
     * with the <|endoftext|> token between them, and so are the validation
     * parts, so that the model does not learn that one book runs on into the
     * next. With a single file no token is added.
     *
     * @param hyperparameters - The source of trainingRatio and trainingWindowLength.
     * @param tokenizer - Converts the text into token ids.
     * @returns The training and validation windows.
     * @throws Error if a file cannot be read.
     */
    load(hyperparameters, tokenizer) {
        // The training and validation parts of each file, as token ids.
        const trainingParts = [];
        const validationParts = [];
        for (const filePath of this.filePaths) {
            const fileText = fs.readFileSync(filePath, "utf-8");
            const text = this.maximumCharacters === undefined ? fileText : fileText.slice(0, this.maximumCharacters);
            const splitIndex = Math.floor(hyperparameters.trainingRatio * text.length);
            trainingParts.push(tokenizer.encode(text.slice(0, splitIndex)));
            validationParts.push(tokenizer.encode(text.slice(splitIndex)));
            if (DEBUG.TRAINER) {
                console.log(`TextDataset| Loaded ${filePath}. Characters used: ${text.length} of ${fileText.length}. ` +
                    `Training tokens: ${trainingParts[trainingParts.length - 1].length}. ` +
                    `Validation tokens: ${validationParts[validationParts.length - 1].length}.`);
            }
        }
        const trainingTokens = joinDocuments(trainingParts, tokenizer.endOfTextId);
        const validationTokens = joinDocuments(validationParts, tokenizer.endOfTextId);
        if (DEBUG.TRAINER && this.filePaths.length > 1) {
            console.log(`TextDataset| ${this.filePaths.length} files loaded. ` +
                `Training tokens: ${trainingTokens.length}. Validation tokens: ${validationTokens.length}.`);
        }
        return {
            training: createWindows(trainingTokens, hyperparameters.trainingWindowLength),
            validation: createWindows(validationTokens, hyperparameters.trainingWindowLength),
        };
    }
}
export default TextDataset;
