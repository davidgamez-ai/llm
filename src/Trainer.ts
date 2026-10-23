import * as fs from "fs";
import BPETokenizer from "./BPETokenizer.js";
import Hyperparameters from "./Hyperparameters.js";
import GPTModel from "./GPTModel.js";
import { crossEntropyLoss, crossEntropyErrors } from "./Loss.js";
import AdamW from "./AdamW.js";
import { readCsvColumn } from "./CsvReader.js";
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

  /**
   * The average loss of the training batches in this epoch, calculated in
   * training mode (with dropout). Each batch's loss is measured during its
   * forward passes, before that batch's weight update.
   */
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
 * - For each window in a batch, the errors at the logits are backpropagated
 *   through the model and its gradients calculated; the gradients are
 *   averaged over the windows of the batch, as in dnn's trainer.
 * - At the end of each batch, the AdamW optimizer updates every weight in
 *   the model once.
 * - At the end of each epoch the validation loss is calculated in
 *   VALIDATION_MODE, so dropout is off.
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
   * The optimizer that updates the model's weights. Created once, with the
   * trainer, so that its running averages carry over from batch to batch,
   * from epoch to epoch, and from one call of train() to the next.
   */
  private readonly optimizer: AdamW;

  /**
   * The number of TinyStories training stories loaded so far by
   * loadTinyStories(). The next call starts after them, so each training
   * run sees new stories.
   */
  private _tinyStoriesLoaded: number = 0;

  /**
   * The number of TinyStories training stories loaded so far by
   * loadTinyStories(), across every call. The next call loads the stories
   * that follow these in the training file.
   */
  get tinyStoriesLoaded(): number {
    return this._tinyStoriesLoaded;
  }

  /**
   * Creates a Trainer for the given model, building its own tokenizer and
   * AdamW optimizer. The model is passed in rather than built here so that
   * the same instance can be shared with a Generator, which then generates
   * text with the trained weights.
   *
   * @param hyperparameters - The settings the model was built with; also the source of contextLength, batchSize, trainingRatio, numberEpochs, the optimizer's settings and the random number stream used for shuffling, and the mode that train() switches between TRAINING_MODE and VALIDATION_MODE.
   * @param model - The model to train. Must have been built from the same hyperparameters.
   */
  constructor(hyperparameters: Hyperparameters, model: GPTModel) {
    this.hyperparameters = hyperparameters;
    this.tokenizer = new BPETokenizer();
    this.model = model;
    this.optimizer = new AdamW(hyperparameters);
  }

  /**
   * Reads one or more text files, splits each into a training part and a
   * validation part, and encodes the parts with the BPE tokenizer, replacing
   * any previously loaded tokens.
   *
   * As in the book, each file's text is split by characters before it is
   * tokenized: its first trainingRatio of the characters are used for
   * training and the rest for validation. Each file is split separately,
   * rather than joining the files and splitting once, so the validation set
   * contains a sample of every file, not just the end of the last one.
   *
   * The training parts of the files are then joined, in the order given,
   * with the <|endoftext|> token between them, and so are the validation
   * parts. This is how GPT-2 marks the boundaries between documents: the
   * model learns that the text after the token is unrelated to the text
   * before it, rather than learning that one book runs on into the next.
   * With a single file no token is added, so the result is the same as
   * loading that file on its own.
   *
   * @param filePaths - Path to a UTF-8 text file, or an array of paths.
   * @throws Error if filePaths is an empty array.
   */
  load(filePaths: string | string[]): void {
    const paths:string[] = typeof filePaths === "string" ? [filePaths] : filePaths;
    if (paths.length === 0) {
      throw new Error("Trainer| load() needs at least one file");
    }

    // The training and validation parts of each file, as token ids.
    const trainingParts:number[][] = [];
    const validationParts:number[][] = [];

    for (const filePath of paths) {
      const text:string = fs.readFileSync(filePath, "utf-8");
      const splitIndex:number = Math.floor(this.hyperparameters.trainingRatio * text.length);
      trainingParts.push(this.tokenizer.encode(text.slice(0, splitIndex)));
      validationParts.push(this.tokenizer.encode(text.slice(splitIndex)));

      if (DEBUG.TRAINER) {
        console.log(
          `Trainer| Loaded ${filePath}. Characters: ${text.length}. ` +
          `Training tokens: ${trainingParts[trainingParts.length - 1].length}. ` +
          `Validation tokens: ${validationParts[validationParts.length - 1].length}.`
        );
      }
    }

    this.trainingTokens = this.joinDocuments(trainingParts);
    this.validationTokens = this.joinDocuments(validationParts);

    if (DEBUG.TRAINER && paths.length > 1) {
      console.log(
        `Trainer| ${paths.length} files loaded. ` +
        `Training tokens: ${this.trainingTokens.length}. Validation tokens: ${this.validationTokens.length}.`
      );
    }
  }

  /**
   * Loads stories from the TinyStories dataset (Eldan and Li, 2023), ready
   * for training in the same way as load(): the training and validation
   * tokens are replaced, and train() then uses them.
   *
   * TinyStories is a large collection of short children's stories, written
   * with a small vocabulary so that small models can learn to write coherent
   * English. It comes as two CSV files with a "text" column holding one story
   * per record. Unlike load(), nothing is split off for validation: the
   * dataset already has a separate validation file.
   *
   * Each call loads the next numberTrainingStories stories of the training
   * file: the first call loads the first ones, and each later call starts
   * where the previous one finished (see tinyStoriesLoaded). This way every
   * training run sees new stories, which teaches the model more than seeing
   * the same ones again. Validation always uses the first
   * numberValidationStories of the validation file, so that the validation
   * losses of different runs can be compared. Only the stories needed are
   * read, so the training file (nearly 2 GB) is never loaded in full. Each
   * story is encoded separately, and the stories are joined with the
   * <|endoftext|> token between them, as load() joins files: the model
   * learns that each story is separate.
   *
   * If a file has fewer stories than requested (for the training file, fewer
   * still unused), all of them are used and a warning is printed.
   *
   * @param trainingFilePath - Path to the TinyStories training CSV file, e.g. data/TinyStories_train.csv.
   * @param validationFilePath - Path to the TinyStories validation CSV file, e.g. data/TinyStories_validation.csv.
   * @throws Error if either file has no "text" column.
   */
  loadTinyStories(trainingFilePath: string, validationFilePath: string): void {
    // The training stories continue from where the last call finished; the validation stories are always the same.
    const firstStory:number = this._tinyStoriesLoaded;
    const trainingStories:string[] = this.readStories(trainingFilePath, firstStory, this.hyperparameters.numberTrainingStories);
    const validationStories:string[] = this.readStories(validationFilePath, 0, this.hyperparameters.numberValidationStories);
    this._tinyStoriesLoaded += trainingStories.length;

    this.trainingTokens = this.joinDocuments(trainingStories.map((story:string) => this.tokenizer.encode(story)));
    this.validationTokens = this.joinDocuments(validationStories.map((story:string) => this.tokenizer.encode(story)));

    if (DEBUG.TRAINER) {
      console.log(
        `Trainer| TinyStories loaded. Training stories: ${trainingStories.length}, ` +
        `numbers ${firstStory + 1} to ${firstStory + trainingStories.length} (${this.trainingTokens.length} tokens). ` +
        `Validation stories: ${validationStories.length} (${this.validationTokens.length} tokens).`
      );
    }
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
  private readStories(filePath: string, skipStories: number, numberStories: number): string[] {
    const stories:string[] = readCsvColumn(filePath, "text", numberStories, skipStories);
    if (stories.length < numberStories) {
      const position:string = skipStories > 0 ? ` after the first ${skipStories}` : "";
      console.warn(`Trainer| ${filePath} has only ${stories.length} stories${position}; ${numberStories} were requested.`);
    }
    return stories;
  }

  /**
   * Joins documents (files or stories), already encoded as token ids, into
   * one list, with the <|endoftext|> token between each document and the
   * next. This is how GPT-2 marks the boundaries between documents: the
   * model learns that the text after the token is unrelated to the text
   * before it. A single document is returned unchanged, with no token added.
   *
   * @param documents - The token ids of each document, in order.
   * @returns The joined token ids.
   */
  private joinDocuments(documents: number[][]): number[] {
    const tokens:number[] = [];
    for (let i:number = 0; i < documents.length; i++) {
      if (i > 0) {
        tokens.push(this.tokenizer.endOfTextId);
      }
      /*
       * Appended with a loop rather than push(...document), because spreading
       * a whole book's tokens as arguments can exceed the engine's argument
       * limit.
       */
      for (const token of documents[i]) {
        tokens.push(token);
      }
    }
    return tokens;
  }

  /**
   * Trains the model for numberEpochs epochs over the training set. In each
   * epoch the training windows are shuffled and grouped into batches of
   * batchSize windows, dropping a final incomplete batch, and the model is
   * trained on each batch in turn in TRAINING_MODE (see trainOnBatch). At
   * the end of the epoch the validation loss is calculated, and both losses
   * are logged.
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
          const batchLoss:number = this.trainOnBatch(batches[batchIndex]);
          totalBatchLoss += batchLoss;

          if (DEBUG.TRAINER) {
            console.log(`Trainer| Epoch ${epoch}. Batch ${batchIndex + 1} of ${batches.length}. Batch loss: ${batchLoss}.`);
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
   * Trains the model on one batch: the heart of training. For each window in
   * the batch:
   *
   * 1. Forward pass: the window's inputs are fed through the model, giving
   *    one row of logits per position.
   * 2. Loss: the cross entropy of each position's logits against the token
   *    that actually comes next, added up to measure how well the model did.
   * 3. Errors: the gradient of the window's loss with respect to each logit,
   *    softmax minus the one-hot target, scaled by 1 / (number of positions)
   *    because the window's loss is the mean over its positions.
   * 4. Backpropagation: the errors are passed back through the model, each
   *    component storing what it needs.
   * 5. Gradients: each component calculates its gradients. The first window
   *    of the batch replaces the previous batch's gradients; the rest are
   *    averaged in, so the result is the gradient of the whole batch's mean
   *    loss.
   *
   * Then, once the whole batch has been processed, the optimizer updates
   * every weight once, using the batch's averaged gradients.
   *
   * Hyperparameters.mode must already be TRAINING_MODE.
   *
   * @param batch - The windows in the batch.
   * @returns The batch's loss: the average loss per token, measured before the weights were updated.
   */
  private trainOnBatch(batch: Window[]): number {
    let totalLoss:number = 0;
    let tokenCount:number = 0;

    for (let windowIndex:number = 0; windowIndex < batch.length; windowIndex++) {
      const window:Window = batch[windowIndex];

      // 1. Forward pass.
      const logits:number[][] = this.model.update(window.inputs);

      // 2 and 3. The loss and the errors at the logits, for every position.
      const scale:number = 1 / window.targets.length;
      const outputErrors:number[][] = [];
      for (let position:number = 0; position < logits.length; position++) {
        totalLoss += crossEntropyLoss(logits[position], window.targets[position]);
        outputErrors.push(crossEntropyErrors(logits[position], window.targets[position], scale));
      }
      tokenCount += window.targets.length;

      // 4. Backpropagation.
      this.model.backpropagate(outputErrors);

      // 5. Gradients: replace them for the first window of the batch, average the rest in.
      const isStartOfBatch:boolean = windowIndex === 0;
      this.model.calculateGradients(!isStartOfBatch);
    }

    // One weight update per batch, from the gradients averaged over its windows.
    this.model.updateWeights(this.optimizer);

    return totalLoss / tokenCount;
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
    const logits:number[][] = this.model.update(inputs);

    let totalLoss:number = 0;
    for (let position:number = 0; position < logits.length; position++) {
      // The negative log probability the model gives to the token that actually comes next.
      totalLoss += crossEntropyLoss(logits[position], targets[position]);

      if (DEBUG.TRAINER && DEBUG.VERBOSE) {
        console.log(`Trainer| Position ${position}. Total loss: ${totalLoss}`);
      }
    }

    return totalLoss;
  }
}

export default Trainer;
