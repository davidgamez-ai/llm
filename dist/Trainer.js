import * as fs from "fs";
import BPETokenizer from "./BPETokenizer.js";
import Hyperparameters from "./Hyperparameters.js";
import GPTModel from "./GPTModel.js";
import DEBUG from "./Debug.js";
/** Loads training data and trains the model on it. */
class Trainer {
    /** The token ids of the most recently loaded training file. Empty until load() is called. */
    tokens = [];
    /**
     * The Hyperparameters this trainer was built with. A reference is kept,
     * rather than copies of its values, so that later changes are seen by the
     * trainer.
     */
    hyperparameters;
    /** Converts the training text into token ids. */
    tokenizer;
    /** The model being trained. */
    model;
    /**
     * Creates a Trainer, building its own tokenizer and model.
     *
     * @param hyperparameters - The settings used to build the model. Defaults to a new Hyperparameters instance.
     */
    constructor(hyperparameters = new Hyperparameters()) {
        this.hyperparameters = hyperparameters;
        this.tokenizer = new BPETokenizer();
        this.model = new GPTModel(hyperparameters);
    }
    /**
     * Reads the given file, encodes its contents using the BPE tokenizer and
     * stores the resulting token ids, replacing any previously loaded tokens.
     *
     * @param filePath - Path to the UTF-8 text file to load.
     */
    load(filePath) {
        const text = fs.readFileSync(filePath, "utf-8");
        this.tokens = this.tokenizer.encode(text);
        if (DEBUG.TRAINER) {
            console.log(`Trainer| File Loaded. Number of characters: ${text.length}. Number of tokens: ${this.tokens.length}.`);
        }
    }
    /**
     * Slides a window of contextLength tokens across the loaded tokens and
     * calculates the average cross entropy loss of the model at each window
     * position, in training mode. The window moves contextLength tokens at a
     * time, so the windows do not overlap and every token is used as an input
     * once. The last window may be shorter if the tokens do not divide evenly.
     *
     * Every batchSize windows form a batch, and the average loss of the batch
     * is logged to the console. If the last batch has fewer than batchSize
     * windows, its average loss is logged as well.
     *
     * Hyperparameters.training is switched on for the calculation and restored
     * to its original value afterwards, even if an error is thrown.
     *
     * @returns The loss averaged across every window.
     * @throws Error if fewer than 2 tokens have been loaded, since there is then no next token to predict.
     */
    train() {
        if (this.tokens.length < 2) {
            throw new Error(`Trainer| At least 2 tokens must be loaded to train; got ${this.tokens.length}`);
        }
        const originalTraining = this.hyperparameters.training;
        this.hyperparameters.training = true;
        if (DEBUG.TRAINER) {
            console.log("Trainer| Training started.");
        }
        try {
            const contextLength = this.hyperparameters.contextLength;
            const batchSize = this.hyperparameters.batchSize;
            let totalLoss = 0;
            let windowCount = 0;
            let batchLoss = 0;
            let batchCounter = 0;
            let batchNumber = 0;
            /*
             * The target for each input token is the token that follows it, so the
             * targets are the inputs shifted one place to the right. The window
             * stops while at least one token remains after its start, so the last
             * input of every window still has a target; the final window is
             * shortened to fit.
             */
            for (let start = 0; start < this.tokens.length - 1; start += contextLength) {
                const inputLength = Math.min(contextLength, this.tokens.length - 1 - start);
                const inputs = this.tokens.slice(start, start + inputLength);
                const targets = this.tokens.slice(start + 1, start + inputLength + 1);
                const windowLoss = this.calculateLoss(inputs, targets);
                if (DEBUG.TRAINER) {
                    console.log(`Trainer| Window starting at token ${start}. Input tokens: ${inputs.length}. Average loss: ${windowLoss}.`);
                }
                totalLoss += windowLoss;
                ++windowCount;
                batchLoss += windowLoss;
                ++batchCounter;
                // A full batch of windows has been processed: log its average loss and start a new batch.
                if (batchCounter === batchSize) {
                    ++batchNumber;
                    console.log(`Trainer| Batch ${batchNumber}. Average batch loss: ${batchLoss / batchCounter}.`);
                    batchLoss = 0;
                    batchCounter = 0;
                }
            }
            // Log the final batch if it ran out of windows before reaching batchSize.
            if (batchCounter > 0) {
                ++batchNumber;
                console.log(`Trainer| Batch ${batchNumber} (partial, ${batchCounter} of ${batchSize} windows). ` +
                    `Average batch loss: ${batchLoss / batchCounter}.`);
            }
            const averageLoss = totalLoss / windowCount;
            if (DEBUG.TRAINER) {
                console.log(`Trainer| Training finished. Windows: ${windowCount}. Average loss: ${averageLoss}.`);
            }
            return averageLoss;
        }
        finally {
            this.hyperparameters.training = originalTraining;
        }
    }
    /**
     * Feeds one window of tokens into the model and calculates the average
     * cross entropy loss: the mean, over every position in the window, of the
     * negative log probability the model gives to the token that actually
     * comes next. Hyperparameters.training must already be true, so the model
     * returns logits for every position.
     *
     * @param inputs - The token ids fed into the model.
     * @param targets - The token id that follows each input, i.e. the inputs shifted one place to the right. Must be the same length as inputs.
     * @returns The loss averaged across the window.
     */
    calculateLoss(inputs, targets) {
        // In training mode the model returns one row of logits per input token.
        const logits = this.model.calculate(inputs);
        let totalLoss = 0;
        logits.forEach((row, position) => {
            /*
             * The model returns logits in training mode, so the log probability of
             * the target is found with a log-softmax: logit[target] minus the log
             * of the sum of the exponentiated logits. The row's maximum is
             * subtracted before exponentiating to avoid overflow; this does not
             * change the result because softmax is unaffected by adding a
             * constant to every input.
             */
            const maxLogit = row.reduce((max, value) => Math.max(max, value), -Infinity);
            const sumExp = row.reduce((sum, value) => sum + Math.exp(value - maxLogit), 0);
            const logProbability = row[targets[position]] - maxLogit - Math.log(sumExp);
            totalLoss -= logProbability;
            if (DEBUG.TRAINER && DEBUG.VERBOSE) {
                console.log(`Trainer| Position ${position}. Total loss: ${totalLoss}`);
            }
        });
        return totalLoss / logits.length;
    }
}
export default Trainer;
