import BPETokenizer from "./BPETokenizer.js";
import Hyperparameters from "./Hyperparameters.js";

/**
 * One example for the model to learn from: a run of token ids, and the token
 * id that follows each of them.
 */
export interface Window {
  /** The token ids fed into the model. */
  inputs: number[];

  /** The token id that follows each input, i.e. the inputs shifted one place to the right. */
  targets: number[];
}

/** The windows a dataset has prepared: what a Trainer trains and validates on. */
export interface DatasetWindows {
  /** The windows to train on. */
  training: Window[];

  /** The windows to measure the validation loss on. */
  validation: Window[];
}

/**
 * A source of training data. Each dataset reads its own format and prepares
 * windows, so that Trainer only has to run its training loop: a new kind of
 * data is a new implementation of this interface, with no change to Trainer.
 * The implementations are TextDataset, TinyStoriesDataset and AlpacaDataset.
 */
export interface Dataset {
  /**
   * Prepares the next windows to train on, using the current settings.
   *
   * @param hyperparameters - The current settings, such as trainingWindowLength.
   * @param tokenizer - Converts the dataset's text into token ids.
   * @returns The training and validation windows.
   */
  load(hyperparameters: Hyperparameters, tokenizer: BPETokenizer): DatasetWindows;
}

/**
 * Cuts tokens into non-overlapping windows of windowLength tokens, as the
 * book's GPTDatasetV1 does with a stride equal to the context length. The
 * targets of each window are its inputs shifted one place to the right, so a
 * window needs windowLength + 1 tokens. Only full windows are made: any tokens
 * left over at the end are not used.
 *
 * @param tokens - The token ids to cut into windows.
 * @param windowLength - The number of inputs in each window.
 * @returns The windows, in the order they appear in tokens.
 */
export function createWindows(tokens: number[], windowLength: number): Window[] {
  const windows: Window[] = [];

  // A window starting at start needs tokens up to start + windowLength for its last target.
  for (let start: number = 0; start + windowLength < tokens.length; start += windowLength) {
    windows.push({
      inputs: tokens.slice(start, start + windowLength),
      targets: tokens.slice(start + 1, start + windowLength + 1),
    });
  }
  return windows;
}

/**
 * Joins documents (files or stories), already encoded as token ids, into one
 * list, with the <|endoftext|> token between each document and the next. This
 * is how GPT-2 marks the boundaries between documents: the model learns that
 * the text after the token is unrelated to the text before it. A single
 * document is returned unchanged, with no token added.
 *
 * @param documents - The token ids of each document, in order.
 * @param endOfTextId - The id of the <|endoftext|> token.
 * @returns The joined token ids.
 */
export function joinDocuments(documents: number[][], endOfTextId: number): number[] {
  const tokens: number[] = [];
  for (let i: number = 0; i < documents.length; i++) {
    if (i > 0) {
      tokens.push(endOfTextId);
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
