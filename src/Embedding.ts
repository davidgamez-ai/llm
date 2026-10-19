import BPETokenizer from "./BPETokenizer.js";
import Hyperparameters from "./Hyperparameters.js";
import { randomNormalMatrix } from "./WeightInitialization.js";
import Matrix from "./Matrix.js";
import DEBUG from "./Debug.js";

/**
 * Two embedding matrices, each mapping an id (a token id, or a sequence
 * position) to a vector of embeddingSize random values. The token embedding
 * matrix has one row per vocabulary token; the position embedding matrix has
 * one row per position in the context window, as in GPT-2. The vocabulary
 * size, context length and embedding size are taken from Hyperparameters, so
 * they always match the values used elsewhere in the application.
 */
class Embedding {
  /** Number of rows in the token embedding matrix, i.e. the number of distinct token ids supported. */
  vocabSize: number;

  /** Number of rows in the position embedding matrix, i.e. the maximum number of tokens in a sequence. */
  contextLength: number;

  /** Number of columns in each embedding matrix, i.e. the width of each embedding vector. */
  embeddingSize: number;

  /**
   * Standard deviation of the normal distribution (mean 0) used to
   * initialize the position embedding matrix. OpenAI's original GPT-2
   * TensorFlow code uses 0.01 for its position embeddings (wpe); Hugging
   * Face's implementation uses 0.02.
   */
  positionEmbeddingStandardDeviation: number;

  /**
   * Standard deviation of the normal distribution (mean 0) used to
   * initialize the token embedding matrix. GPT-2 uses 0.02 for its token
   * embeddings (wte).
   */
  tokenEmbeddingStandardDeviation: number;

  // The token embedding matrix built by build(): one row per vocabulary
  // token, each row embeddingSize values wide. Looked up by token id.
  private textEmbeddingMatrix: Matrix = new Matrix(0, 0);

  // The position embedding matrix built by build(): one row per position in
  // the context window (contextLength rows), each row embeddingSize values
  // wide. Looked up by a token's position in the sequence rather than by its id.
  private positionEmbeddingMatrix: Matrix = new Matrix(0, 0);

  /** Draws from the Hyperparameters' shared random number stream, which is seeded when Hyperparameters.seed is set. */
  private readonly random: () => number;

  /**
   * Creates an Embedding, copying vocabSize, contextLength, embeddingSize,
   * positionEmbeddingStandardDeviation and tokenEmbeddingStandardDeviation
   * from the given Hyperparameters so this instance always matches the
   * values used elsewhere in the application.
   *
   * @param hyperparameters - Source of vocabularySize, contextLength, embeddingSize, positionEmbeddingStandardDeviation, tokenEmbeddingStandardDeviation and the random number stream. Defaults to a new Hyperparameters instance.
   */
  constructor(hyperparameters: Hyperparameters = new Hyperparameters()) {
    this.vocabSize = hyperparameters.vocabularySize;
    this.contextLength = hyperparameters.contextLength;
    this.embeddingSize = hyperparameters.embeddingSize;
    this.positionEmbeddingStandardDeviation = hyperparameters.positionEmbeddingStandardDeviation;
    this.tokenEmbeddingStandardDeviation = hyperparameters.tokenEmbeddingStandardDeviation;
    this.random = () => hyperparameters.random();
  }

  /**
   * Builds the two embedding matrices, stores them on the instance, and logs
   * their dimensions:
   * - textEmbeddingMatrix: vocabSize x embeddingSize, initialized as in
   *   GPT-2 (Hugging Face GPT2PreTrainedModel._init_weights), with values drawn from a
   *   normal distribution with mean 0 and standard deviation
   *   tokenEmbeddingStandardDeviation (0.02 by default).
   * - positionEmbeddingMatrix: contextLength x embeddingSize, initialized as
   *   in OpenAI's original GPT-2 TensorFlow code (model.py, wpe), with values
   *   drawn from a normal distribution with mean 0 and standard deviation
   *   positionEmbeddingStandardDeviation (0.01 by default). Hugging Face's
   *   GPT2PreTrainedModel._init_weights uses 0.02 for the position embeddings instead,
   *   the same as for the token embeddings.
   *
   * Random values come from the Hyperparameters' shared stream, so the
   * matrices are reproducible when Hyperparameters.seed is set.
   *
   * @returns void. The resulting matrices are stored on the instance for use by getEmbedding.
   */
  build(): void {
    this.textEmbeddingMatrix = randomNormalMatrix(
      this.vocabSize, this.embeddingSize, this.tokenEmbeddingStandardDeviation, this.random
    );
    this.positionEmbeddingMatrix = randomNormalMatrix(
      this.contextLength, this.embeddingSize, this.positionEmbeddingStandardDeviation, this.random
    );

    if(DEBUG.EMBEDDING) {
      console.log(
        `Embeddings| Token embedding matrix built. Rows: ${this.textEmbeddingMatrix.rows}; columns: ${this.textEmbeddingMatrix.columns}`
      );
      console.log(
        `Embeddings| Position embedding matrix built. Rows: ${this.positionEmbeddingMatrix.rows}; columns: ${this.positionEmbeddingMatrix.columns}`
      );
    }
  }

  /**
   * Tokenizes text with BPETokenizer and returns the combined embeddings of
   * the resulting tokens, as calculated by getEmbeddingFromTokens.
   *
   * @param text - Raw input text to embed.
   * @returns One combined embedding vector per token, in token order. Each vector is the token's textEmbeddingMatrix row plus its positionEmbeddingMatrix row.
   */
  getEmbedding(text: string): number[][] {
    return this.getEmbeddingFromTokens(new BPETokenizer().encode(text));
  }

  /**
   * For each token id, looks up its row in textEmbeddingMatrix by token id,
   * and its row in positionEmbeddingMatrix by the token's position in the
   * sequence, and adds the two rows together. Each stage (token ids, token
   * embeddings, position embeddings, combined embeddings) is stored in its
   * own array and logged so intermediate results can be inspected, rather
   * than only returning the final combined result.
   *
   * @param tokenIds - Token ids of already tokenized text, in sequence order.
   * @returns One combined embedding vector per token, in token order. Each vector is the token's textEmbeddingMatrix row plus its positionEmbeddingMatrix row.
   * @throws Error if there are more tokens than contextLength, since the position embedding matrix has no row for positions beyond the context window.
   * @throws RangeError if any token id is not a whole number from 0 to vocabSize - 1, since the token embedding matrix has no row for it.
   */
  getEmbeddingFromTokens(tokenIds: number[]): number[][] {
    if (tokenIds.length > this.contextLength) {
      throw new Error(
        `Embedding| Sequence of ${tokenIds.length} tokens exceeds the context length of ${this.contextLength}`
      );
    }

    /*
     * Check every id before looking any up. An id outside the vocabulary
     * (or a fraction or NaN) has no row in textEmbeddingMatrix, so the
     * lookup would return undefined and fail later with a confusing
     * TypeError, far from the cause.
     */
    for (let position = 0; position < tokenIds.length; position++) {
      const id = tokenIds[position];
      if (!(Number.isInteger(id) && id >= 0 && id < this.vocabSize)) {
        throw new RangeError(
          `Embedding| Token id ${id} at position ${position} is outside the vocabulary (0 to ${this.vocabSize - 1})`
        );
      }
    }

    // The token and position embedding rows are views onto the matrices, not copies.
    const tokenEmbeddings: Float32Array[] = [];
    const positionEmbeddings: Float32Array[] = [];
    const combinedEmbeddings: number[][] = [];

    for (let position = 0; position < tokenIds.length; position++) {
      // Look up the current token's row in the text embedding matrix, by id.
      const id = tokenIds[position];
      const tokenEmbedding = this.textEmbeddingMatrix.getRow(id);
      tokenEmbeddings.push(tokenEmbedding);

      // Look up this token's row in the position embedding matrix, by its
      // position in the sequence rather than by its id.
      const positionEmbedding = this.positionEmbeddingMatrix.getRow(position);
      positionEmbeddings.push(positionEmbedding);

      // Combine the token embedding and position embedding element-wise so
      // the returned vector encodes both the token's identity and its
      // position in the sequence.
      const combinedEmbedding: number[] = [];
      for (let i = 0; i < this.embeddingSize; i++) {
        combinedEmbedding.push(tokenEmbedding[i] + positionEmbedding[i]);
      }
      combinedEmbeddings.push(combinedEmbedding);
    }

    // Log every stage, not just the final result, so each step of the
    // calculation can be checked independently.
    if(DEBUG.EMBEDDING) {
      this.logArray("Token ids", tokenIds);
      this.logArray("Token embeddings", tokenEmbeddings);
      this.logArray("Position embeddings", positionEmbeddings);
      this.logArray("Combined embeddings", combinedEmbeddings);
    }

    return combinedEmbeddings;
  }

  /**
   * Counts the trainable parameters in the embedding: every value in the
   * token embedding matrix and the position embedding matrix. Returns 0
   * until build() has been called.
   *
   * @returns The total number of values currently stored in textEmbeddingMatrix and positionEmbeddingMatrix.
   */
  getParameterCount(): number {
    return this.textEmbeddingMatrix.size + this.positionEmbeddingMatrix.size;
  }

  /**
   * Logs a one or two dimensional array. When DEBUG.VERBOSE is true the
   * whole array is printed with console.table; otherwise only its
   * dimensions are printed, which keeps the output short for large arrays.
   *
   * @param label - Name of the array, printed before its contents or dimensions.
   * @param array - The array to log: a vector, or one row per token. Rows may be Float32Arrays (rows of a Matrix).
   */
  private logArray(label: string, array: number[] | ArrayLike<number>[]): void {
    const isTwoDimensional: boolean = typeof array[0] === "object";
    if (DEBUG.VERBOSE) {
      console.log(`Embedding| ${label}:`);
      // Convert Float32Array rows to plain arrays so console.table lays them out as a grid.
      console.table(isTwoDimensional ? (array as ArrayLike<number>[]).map((row) => Array.from(row)) : array);
    } else if (isTwoDimensional) {
      console.log(`Embedding| ${label}. Rows: ${array.length}; columns: ${(array[0] as ArrayLike<number>).length}`);
    } else {
      console.log(`Embedding| ${label}. Length: ${array.length}`);
    }
  }
}

export default Embedding;
