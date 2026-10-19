import * as fs from "fs";
import BPETokenizer from "./BPETokenizer.js";
import Hyperparameters from "./Hyperparameters.js";
import GPTModel from "./GPTModel.js";
import DEBUG from "./Debug.js";

/**
 * One example for the model to learn from: a run of contextLength token ids,
 * and the token id that follows each of them.
 */
interface Window {
  /** The token ids fed into the model. */
  inputs: number[];

  /** The token id that follows each input, i.e. the inputs shifted one place to the right. */
  targets: number[];
}

/** The losses measured in one epoch of training. */
export interface EpochLoss {
  /** The epoch number, starting from 1. */
  epoch: number;

  /** The average loss of the training batches in this epoch, calculated in training mode (with dropout). */
  trainingLoss: number;

  /** The loss on the validation set at the end of this epoch, calculated in validation mode (without dropout). */
  validationLoss: number;
}

/**
 * Loads training data and trains the model on it, following the book's
 * GPTDatasetV1, DataLoader and train_model_simple:
 *
 * - The text is split into a training set and a validation set
 *   (Hyperparameters.trainingRatio, 90/10 by default).
 * - Each set is cut into non-overlapping windows of contextLength tokens.
 * - In each epoch the training windows are shuffled and grouped into batches
 *   of batchSize windows; a final incomplete batch is dropped.
 * - The loss of each batch is averaged over every token in the batch.
 * - At the end of each epoch the validation loss is calculated in
 *   VALIDATION_MODE, so dropout is off.
 *
 * Weights are not updated yet: that needs backpropagation, which has not
 * been written. Until then the losses show how well the untrained model
 * predicts the next token.
 */
class Trainer {
  /** The token ids of the training part of the most recently loaded file. Empty until load() is called. */
  private trainingTokens: number[] = [];

  /** The token ids of the validation part of the most recently loaded file. Empty until load() is called. */
  private validationTokens: number[] = [];

  /**
   * The Hyperparameters this trainer was built with. A reference is kept,
   * rather than copies of its values, so that later changes are seen by the
   * trainer.
   */
  private readonly hyperparameters: Hyperparameters;

  /** Converts the training text into token ids. */
  private readonly tokenizer: BPETokenizer;

  /** The model being trained. */
  private readonly model: GPTModel;

  /**
   * Creates a Trainer for the given model, building its own tokenizer. The
   * model is passed in rather than built here so that the same instance can
   * be shared with a Generator, which then generates text with the trained
   * weights.
   *
   * @param hyperparameters - The settings the model was built with; also the source of contextLength, batchSize, trainingRatio, numberEpochs and the random number stream used for shuffling, and the mode that train() switches between TRAINING_MODE and VALIDATION_MODE.
   * @param model - The model to train. Must have been built from the same hyperparameters.
   */
  constructor(hyperparameters: Hyperparameters, model: GPTModel) {
    this.hyperparameters = hyperparameters;
    this.tokenizer = new BPETokenizer();
    this.model = model;
  }

  /**
   * Reads the given file, splits its text into a training part and a
   * validation part, and encodes each part with the BPE tokenizer, replacing
   * any previously loaded tokens.
   *
   * As in the book, the text is split by characters before it is tokenized:
   * the first trainingRatio of the characters are used for training and the
   * rest for validation.
   *
   * @param filePath - Path to the UTF-8 text file to load.
   */
  load(filePath: string): void {
    const text:string = fs.readFileSync(filePath, "utf-8");

    const splitIndex:number = Math.floor(this.hyperparameters.trainingRatio * text.length);
    this.trainingTokens = this.tokenizer.encode(text.slice(0, splitIndex));
    this.validationTokens = this.tokenizer.encode(text.slice(splitIndex));

    if (DEBUG.TRAINER) {
      console.log(
        `Trainer| File loaded. Characters: ${text.length}. ` +
        `Training tokens: ${this.trainingTokens.length}. Validation tokens: ${this.validationTokens.length}.`
      );
    }
  }

  /**
   * Runs numberEpochs epochs over the training set. In each epoch the
   * training windows are shuffled and grouped into batches of batchSize
   * windows, dropping a final incomplete batch, and the loss of each batch is
   * calculated in TRAINING_MODE. At the end of the epoch the validation loss
   * is calculated, and both losses are logged.
   *
   * Hyperparameters.mode is changed while training runs and restored to its
   * original value afterwards, even if an error is thrown.
   *
   * @returns The training and validation loss of each epoch, in order.
   * @throws Error if the training set does not contain at least one full batch of windows.
   * @throws Error if the validation set does not contain at least one window.
   */
  train(): EpochLoss[] {
    const trainingWindows:Window[] = this.createWindows(this.trainingTokens);
    const batchSize:number = this.hyperparameters.batchSize;
    if (trainingWindows.length < batchSize) {
      throw new Error(
        `Trainer| The training set has ${trainingWindows.length} windows of ${this.hyperparameters.contextLength} tokens; ` +
        `at least batchSize (${batchSize}) are needed. Load more text or reduce contextLength or batchSize.`
      );
    }
    this.checkValidationSet();

    const originalMode:number = this.hyperparameters.mode;
    const epochLosses:EpochLoss[] = [];

    try {
      for (let epoch:number = 1; epoch <= this.hyperparameters.numberEpochs; ++epoch) {
        this.hyperparameters.mode = Hyperparameters.TRAINING_MODE;

        // Shuffle the windows every epoch so the batches differ, as the book's training DataLoader does.
        const batches:Window[][] = this.createBatches(this.shuffle(trainingWindows), true);

        let totalBatchLoss:number = 0;
        for (let batchIndex:number = 0; batchIndex < batches.length; batchIndex++) {
          const batchLoss:number = this.calculateBatchLoss(batches[batchIndex]);
          totalBatchLoss += batchLoss;

          /*
           * Once backpropagation has been written, the gradients of batchLoss
           * will be calculated and the weights updated here, once per batch.
           */

          if (DEBUG.TRAINER) {
            console.log(`Trainer| Epoch ${epoch}. Batch ${batchIndex + 1} of ${batches.length}. Loss: ${batchLoss}.`);
          }
        }

        const epochLoss:EpochLoss = {
          epoch,
          trainingLoss: totalBatchLoss / batches.length,
          validationLoss: this.calculateValidationLoss(),
        };
        epochLosses.push(epochLoss);

        console.log(
          `Trainer| Epoch ${epoch} of ${this.hyperparameters.numberEpochs}. ` +
          `Training loss: ${epochLoss.trainingLoss}. Validation loss: ${epochLoss.validationLoss}.`
        );
      }

      return epochLosses;
    } finally {
      this.hyperparameters.mode = originalMode;
    }
  }

  /**
   * Calculates the loss of the model on the validation set in
   * VALIDATION_MODE, so dropout is off and the result is deterministic. As
   * with the book's validation DataLoader, the windows are not shuffled and
   * a final incomplete batch is kept. The result is the average of the batch
   * losses, as in the book's calc_loss_loader.
   *
   * Hyperparameters.mode is restored to its original value afterwards, even
   * if an error is thrown.
   *
   * @returns The average validation loss.
   * @throws Error if the validation set does not contain at least one window.
   */
  calculateValidationLoss(): number {
    this.checkValidationSet();

    const originalMode:number = this.hyperparameters.mode;
    this.hyperparameters.mode = Hyperparameters.VALIDATION_MODE;

    try {
      const batches:Window[][] = this.createBatches(this.createWindows(this.validationTokens), false);
      const totalLoss:number = batches.reduce((sum:number, batch:Window[]) => sum + this.calculateBatchLoss(batch), 0);
      return totalLoss / batches.length;
    } finally {
      this.hyperparameters.mode = originalMode;
    }
  }

  /**
   * Checks that the validation set is long enough to give at least one window.
   *
   * @throws Error if the validation set does not contain at least one window.
   */
  private checkValidationSet(): void {
    if (this.createWindows(this.validationTokens).length === 0) {
      throw new Error(
        `Trainer| The validation set has ${this.validationTokens.length} tokens; ` +
        `at least contextLength + 1 (${this.hyperparameters.contextLength + 1}) are needed. ` +
        `Load more text, reduce trainingRatio or reduce contextLength.`
      );
    }
  }

  /**
   * Cuts tokens into non-overlapping windows of contextLength tokens, as the
   * book's GPTDatasetV1 does with a stride equal to the context length. The
   * targets of each window are its inputs shifted one place to the right, so
   * a window needs contextLength + 1 tokens. Only full windows are made: any
   * tokens left over at the end are not used.
   *
   * @param tokens - The token ids to cut into windows.
   * @returns The windows, in the order they appear in tokens.
   */
  private createWindows(tokens: number[]): Window[] {
    const contextLength:number = this.hyperparameters.contextLength;
    const windows:Window[] = [];

    // A window starting at start needs tokens up to start + contextLength for its last target.
    for (let start:number = 0; start + contextLength < tokens.length; start += contextLength) {
      windows.push({
        inputs: tokens.slice(start, start + contextLength),
        targets: tokens.slice(start + 1, start + contextLength + 1),
      });
    }
    return windows;
  }

  /**
   * Returns a shuffled copy of the windows, using the Fisher-Yates shuffle.
   * Random numbers come from the Hyperparameters' shared stream, so the
   * order is reproducible when Hyperparameters.seed is set.
   *
   * @param windows - The windows to shuffle. Not changed.
   * @returns A new array holding the same windows in a random order.
   */
  private shuffle(windows: Window[]): Window[] {
    const shuffled:Window[] = [...windows];

    // Work backwards, swapping each window with a randomly chosen one at or before it.
    for (let i:number = shuffled.length - 1; i > 0; --i) {
      const j:number = Math.floor(this.hyperparameters.random() * (i + 1));
      [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
    }
    return shuffled;
  }

  /**
   * Groups windows into batches of batchSize windows, in order.
   *
   * @param windows - The windows to group.
   * @param dropLast - True to drop a final batch with fewer than batchSize windows, as the book does for training; false to keep it, as the book does for validation.
   * @returns The batches.
   */
  private createBatches(windows: Window[], dropLast: boolean): Window[][] {
    const batchSize:number = this.hyperparameters.batchSize;
    const batches:Window[][] = [];

    for (let start:number = 0; start < windows.length; start += batchSize) {
      const batch:Window[] = windows.slice(start, start + batchSize);
      if (batch.length === batchSize || !dropLast) {
        batches.push(batch);
      }
    }
    return batches;
  }

  /**
   * Calculates the loss of one batch: the cross entropy averaged over every
   * token in every window of the batch, as the book's calc_loss_batch does by
   * flattening the batch before calling cross_entropy.
   *
   * @param batch - The windows in the batch.
   * @returns The average loss per token in the batch.
   */
  private calculateBatchLoss(batch: Window[]): number {
    let totalLoss:number = 0;
    let tokenCount:number = 0;
    for (const window of batch) {
      totalLoss += this.calculateTotalLoss(window.inputs, window.targets);
      tokenCount += window.targets.length;
    }
    return totalLoss / tokenCount;
  }

  /**
   * Feeds one window of tokens into the model and calculates the total cross
   * entropy loss: the sum, over every position in the window, of the
   * negative log probability the model gives to the token that actually
   * comes next. Hyperparameters.mode must already be TRAINING_MODE or
   * VALIDATION_MODE, so the model returns logits for every position.
   *
   * @param inputs - The token ids fed into the model.
   * @param targets - The token id that follows each input, i.e. the inputs shifted one place to the right. Must be the same length as inputs.
   * @returns The loss summed across the window.
   */
  private calculateTotalLoss(inputs: number[], targets: number[]): number {
    // In training and validation modes the model returns one row of logits per input token.
    const logits:number[][] = this.model.calculate(inputs);

    let totalLoss:number = 0;
    for (let position:number = 0; position < logits.length; position++) {
      const row:number[] = logits[position];

      /*
       * The model returns logits rather than probabilities, so the log probability of
       * the target is found with a log-softmax: logit[target] minus the log
       * of the sum of the exponentiated logits. The row's maximum is
       * subtracted before exponentiating to avoid overflow; this does not
       * change the result because softmax is unaffected by adding a
       * constant to every input.
       */
      const maxLogit:number = row.reduce((max:number, value:number) => Math.max(max, value), -Infinity);
      const sumExp:number = row.reduce((sum:number, value:number) => sum + Math.exp(value - maxLogit), 0);
      const logProbability:number = row[targets[position]] - maxLogit - Math.log(sumExp);
      totalLoss -= logProbability;

      if (DEBUG.TRAINER && DEBUG.VERBOSE) {
        console.log(`Trainer| Position ${position}. Total loss: ${totalLoss}`);
      }
    }

    return totalLoss;
  }
}

export default Trainer;
