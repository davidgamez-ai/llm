import BPETokenizer from "./BPETokenizer.js";

/**
 * Central store for hyperparameters shared across multiple parts of the
 * application (e.g. Embedding), so that a single instance can be passed
 * around instead of each consumer redeclaring its own copies of the same
 * values.
 */
class Hyperparameters {
  private readonly _embeddingSize: number;
  private readonly _contextLength: number;
  private readonly _vocabularySize: number;
  private readonly _weightMatrixColumns: number;
  private readonly _dropoutRate: number;
  private readonly _training: boolean;
  private readonly _numberAttentionHeads: number;
  private readonly _numberTransformerBlocks: number;

  /**
   * Creates a Hyperparameters instance.
   *
   * @param embeddingSize - The size of each token embedding vector.
   * @param contextLength - The number of tokens of context the model
   * operates over at once.
   * @param weightMatrixColumns - The number of columns in each of the
   * query/key/value weight matrices, i.e. the size of the projected
   * vectors those matrices produce.
   * @param dropoutRate - The probability, between 0 and 1 inclusive, that
   * any given value is dropped (set to zero) during dropout.
   * @param training - True for training mode, in which dropout is applied;
   * false for inference mode, in which dropout is skipped.
   * @param numberAttentionHeads - The number of attention heads used in
   * multi-head attention. Must be a positive whole number.
   * @param numberTransformerBlocks - The number of transformer blocks stacked
   * in the model. Must be a positive whole number.
   * @throws RangeError if dropoutRate is less than 0, greater than 1 or NaN.
   * @throws RangeError if numberAttentionHeads is not a positive whole number.
   * @throws RangeError if numberTransformerBlocks is not a positive whole number.
   */
  constructor(
    embeddingSize: number = 3,
    contextLength: number = 1024,
    weightMatrixColumns: number = 2,
    dropoutRate: number = 0.0,
    training: boolean = true,
    numberAttentionHeads: number = 5,
    numberTransformerBlocks: number = 12
  ) {
    // Written as a negated range check so that NaN, which fails every
    // comparison, is rejected along with values outside [0, 1].
    if (!(dropoutRate >= 0 && dropoutRate <= 1)) {
      throw new RangeError(`Hyperparameters| dropoutRate must be between 0 and 1; got ${dropoutRate}`);
    }

    // Number.isInteger rejects fractions, NaN and Infinity, so combined
    // with the > 0 check only positive whole numbers are accepted.
    if (!(Number.isInteger(numberAttentionHeads) && numberAttentionHeads > 0)) {
      throw new RangeError(
        `Hyperparameters| numberAttentionHeads must be a positive whole number; got ${numberAttentionHeads}`
      );
    }

    // Same positive whole number check as numberAttentionHeads.
    if (!(Number.isInteger(numberTransformerBlocks) && numberTransformerBlocks > 0)) {
      throw new RangeError(
        `Hyperparameters| numberTransformerBlocks must be a positive whole number; got ${numberTransformerBlocks}`
      );
    }

    this._embeddingSize = embeddingSize;
    this._contextLength = contextLength;
    this._vocabularySize = new BPETokenizer().vocabularySize;
    this._weightMatrixColumns = weightMatrixColumns;
    this._dropoutRate = dropoutRate;
    this._training = training;
    this._numberAttentionHeads = numberAttentionHeads;
    this._numberTransformerBlocks = numberTransformerBlocks;
  }

  /** The size of each token embedding vector. */
  get embeddingSize(): number {
    return this._embeddingSize;
  }

  /** The number of tokens of context the model operates over at once. */
  get contextLength(): number {
    return this._contextLength;
  }

  /**
   * The number of columns in each of the query/key/value weight matrices,
   * i.e. the size of the projected vectors those matrices produce.
   */
  get weightMatrixColumns(): number {
    return this._weightMatrixColumns;
  }

  /**
   * The probability, between 0 and 1 inclusive, that any given value is
   * dropped (set to zero) during dropout.
   */
  get dropoutRate(): number {
    return this._dropoutRate;
  }

  /**
   * True for training mode, in which dropout is applied; false for
   * inference mode, in which dropout is skipped.
   */
  get training(): boolean {
    return this._training;
  }

  /** The number of attention heads used in multi-head attention. */
  get numberAttentionHeads(): number {
    return this._numberAttentionHeads;
  }

  /** The number of transformer blocks stacked in the model. */
  get numberTransformerBlocks(): number {
    return this._numberTransformerBlocks;
  }

  /**
   * The size of the vocabulary, loaded from BPETokenizer in the
   * constructor so it always matches the tokenizer actually in use.
   */
  get vocabularySize(): number {
    return this._vocabularySize;
  }
}

export default Hyperparameters;
