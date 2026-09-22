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

  /**
   * Creates a Hyperparameters instance.
   *
   * @param embeddingSize - The size of each token embedding vector.
   * @param contextLength - The number of tokens of context the model
   * operates over at once.
   * @param weightMatrixColumns - The number of columns in each of the
   * query/key/value weight matrices, i.e. the size of the projected
   * vectors those matrices produce.
   */
  constructor(embeddingSize: number = 3, contextLength: number = 1024, weightMatrixColumns: number = 2) {
    this._embeddingSize = embeddingSize;
    this._contextLength = contextLength;
    this._vocabularySize = new BPETokenizer().vocabularySize;
    this._weightMatrixColumns = weightMatrixColumns;
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
   * The size of the vocabulary, loaded from BPETokenizer in the
   * constructor so it always matches the tokenizer actually in use.
   */
  get vocabularySize(): number {
    return this._vocabularySize;
  }
}

export default Hyperparameters;
