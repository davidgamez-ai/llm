import { readCsvColumn } from "./CsvReader.js";
import { createWindows, joinDocuments } from "./Dataset.js";
import DEBUG from "./Debug.js";
/**
 * The TinyStories dataset (Eldan and Li, 2023): a large collection of short
 * children's stories, written with a small vocabulary so that small models can
 * learn to write coherent English. It comes as two CSV files with a "text"
 * column holding one story per record. Nothing is split off for validation,
 * because the dataset already has a separate validation file.
 *
 * Each dataset object keeps its own position in the training file, so a new
 * object starts again from the first story.
 */
class TinyStoriesDataset {
    /** Path to the training CSV file. */
    trainingFilePath;
    /** Path to the validation CSV file. */
    validationFilePath;
    /** The number of training stories loaded so far by load(). The next call starts after them. */
    _storiesLoaded = 0;
    /**
     * Creates a dataset for the given files. Nothing is read until load() is
     * called.
     *
     * @param trainingFilePath - Path to the TinyStories training CSV file, e.g. data/TinyStories_train.csv.
     * @param validationFilePath - Path to the TinyStories validation CSV file, e.g. data/TinyStories_validation.csv.
     */
    constructor(trainingFilePath, validationFilePath) {
        this.trainingFilePath = trainingFilePath;
        this.validationFilePath = validationFilePath;
    }
    /**
     * The number of training stories loaded so far, across every call of
     * load(). The next call loads the stories that follow these in the training
     * file.
     */
    get storiesLoaded() {
        return this._storiesLoaded;
    }
    /**
     * Loads stories and cuts them into windows of trainingWindowLength tokens.
     *
     * Each call loads the next numberTrainingExamples stories of the training
     * file: the first call loads the first ones, and each later call starts
     * where the previous one finished (see storiesLoaded). This way every
     * training run sees new stories, which teaches the model more than seeing
     * the same ones again. Validation always uses the first
     * numberValidationExamples of the validation file, so that the validation
     * losses of different runs can be compared. Only the stories needed are
     * read, so the training file (nearly 2 GB) is never loaded in full. Each
     * story is encoded separately, and the stories are joined with the
     * <|endoftext|> token between them, so that the model learns that each story
     * is separate.
     *
     * If a file has fewer stories than requested (for the training file, fewer
     * still unused), all of them are used and a warning is printed.
     *
     * @param hyperparameters - The source of numberTrainingExamples, numberValidationExamples and trainingWindowLength.
     * @param tokenizer - Converts the stories into token ids.
     * @returns The training and validation windows.
     * @throws Error if either file has no "text" column.
     */
    load(hyperparameters, tokenizer) {
        // The training stories continue from where the last call finished; the validation stories are always the same.
        const firstStory = this._storiesLoaded;
        const trainingStories = this.readStories(this.trainingFilePath, firstStory, hyperparameters.numberTrainingExamples);
        const validationStories = this.readStories(this.validationFilePath, 0, hyperparameters.numberValidationExamples);
        this._storiesLoaded += trainingStories.length;
        const trainingTokens = joinDocuments(trainingStories.map((story) => tokenizer.encode(story)), tokenizer.endOfTextId);
        const validationTokens = joinDocuments(validationStories.map((story) => tokenizer.encode(story)), tokenizer.endOfTextId);
        if (DEBUG.TRAINER) {
            console.log(`TinyStoriesDataset| Training stories: ${trainingStories.length}, ` +
                `numbers ${firstStory + 1} to ${firstStory + trainingStories.length} (${trainingTokens.length} tokens). ` +
                `Validation stories: ${validationStories.length} (${validationTokens.length} tokens).`);
        }
        return {
            training: createWindows(trainingTokens, hyperparameters.trainingWindowLength),
            validation: createWindows(validationTokens, hyperparameters.trainingWindowLength),
        };
    }
    /**
     * Reads stories from a TinyStories CSV file, warning if it has fewer than
     * requested.
     *
     * @param filePath - Path to the CSV file.
     * @param skipStories - The number of stories at the start of the file to skip.
     * @param numberStories - The number of stories to read after them.
     * @returns The stories, in file order.
     */
    readStories(filePath, skipStories, numberStories) {
        const stories = readCsvColumn(filePath, "text", numberStories, skipStories);
        if (stories.length < numberStories) {
            const position = skipStories > 0 ? ` after the first ${skipStories}` : "";
            console.warn(`TinyStoriesDataset| ${filePath} has only ${stories.length} stories${position}; ${numberStories} were requested.`);
        }
        return stories;
    }
}
export default TinyStoriesDataset;
