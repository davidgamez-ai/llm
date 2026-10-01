import BPETokenizer from "./BPETokenizer.js";
import Hyperparameters from "./Hyperparameters.js";
import DEBUG from "./Debug.js";
import createSeededRandom from "./SeededRandom.js";

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
   * initialize the position embedding matrix. GPT-2's original
   * implementation uses 0.01 for its position embeddings (wpe).
   */
  positionEmbeddingStandardDeviation: number;

  // The token embedding matrix built by build(): one row per vocabulary
  // token, each row embeddingSize values wide. Looked up by token id.
  private textEmbeddingMatrix: number[][] = [];

  // The position embedding matrix built by build(): one row per position in
  // the context window (contextLength rows), each row embeddingSize values
  // wide. Looked up by a token's position in the sequence rather than by its id.
  private positionEmbeddingMatrix: number[][] = [];

  /**
   * Creates an Embedding, copying vocabSize, contextLength, embeddingSize
   * and positionEmbeddingStandardDeviation from the given Hyperparameters so
   * this instance always matches the values used elsewhere in the
   * application.
   *
   * @param hyperparameters - Source of vocabularySize, contextLength, embeddingSize and positionEmbeddingStandardDeviation. Defaults to a new Hyperparameters instance.
   */
  constructor(hyperparameters: Hyperparameters = new Hyperparameters()) {
    this.vocabSize = hyperparameters.vocabularySize;
    this.contextLength = hyperparameters.contextLength;
    this.embeddingSize = hyperparameters.embeddingSize;
    this.positionEmbeddingStandardDeviation = hyperparameters.positionEmbeddingStandardDeviation;
  }

  /**
   * Builds the two embedding matrices, stores them on the instance, and logs
   * their dimensions:
   * - textEmbeddingMatrix: vocabSize x embeddingSize, filled with values
   *   drawn uniformly from [min, max).
   * - positionEmbeddingMatrix: contextLength x embeddingSize, initialized as
   *   in GPT-2, with values drawn from a normal distribution with mean 0 and
   *   standard deviation positionEmbeddingStandardDeviation (0.01 by default).
   *
   * @param seed - Optional PRNG seed. When provided, the same seed always produces the same matrices, since Math.random() cannot be seeded and would otherwise make results unreproducible between runs. When omitted, Math.random() is used and the matrices differ on every call.
   * @param min - Inclusive lower bound of the token embedding random range. Defaults to -3.
   * @param max - Exclusive upper bound of the token embedding random range. Defaults to 3.
   * @returns void. The resulting matrices are stored on the instance for use by getEmbedding.
   */
  build(seed?: number, min: number = -3, max: number = 3): void {
    const random = seed === undefined ? Math.random : createSeededRandom(seed);

    /*
     * Draws one sample from a normal distribution with mean 0 and the given
     * standard deviation, using the Box-Muller transform to turn two uniform
     * samples into a standard normal sample. 1 - random() lies in (0, 1], so
     * Math.log never receives 0.
     */
    const randomNormal = (standardDeviation: number): number => {
      const u1 = 1 - random();
      const u2 = random();
      return standardDeviation * Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    };

    this.textEmbeddingMatrix = Array.from({ length: this.vocabSize }, () =>
      Array.from({ length: this.embeddingSize }, () => min + random() * (max - min))
    );

    this.positionEmbeddingMatrix = Array.from({ length: this.contextLength }, () =>
      Array.from({ length: this.embeddingSize }, () => randomNormal(this.positionEmbeddingStandardDeviation))
    );

    if(DEBUG.EMBEDDING) {
      console.log(
        `Embeddings| Token embedding matrix built. Rows: ${this.textEmbeddingMatrix.length}; columns: ${this.textEmbeddingMatrix[0].length}`
      );
      console.log(
        `Embeddings| Position embedding matrix built. Rows: ${this.positionEmbeddingMatrix.length}; columns: ${this.positionEmbeddingMatrix[0].length}`
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
   */
  getEmbeddingFromTokens(tokenIds: number[]): number[][] {
    if (tokenIds.length > this.contextLength) {
      throw new Error(
        `Embedding| Sequence of ${tokenIds.length} tokens exceeds the context length of ${this.contextLength}`
      );
    }

    const tokenEmbeddings: number[][] = [];
    const positionEmbeddings: number[][] = [];
    const combinedEmbeddings: number[][] = [];

    for (let position = 0; position < tokenIds.length; position++) {
      // Look up the current token's row in the text embedding matrix, by id.
      const id = tokenIds[position];
      const tokenEmbedding = this.textEmbeddingMatrix[id];
      tokenEmbeddings.push(tokenEmbedding);

      // Look up this token's row in the position embedding matrix, by its
      // position in the sequence rather than by its id.
      const positionEmbedding = this.positionEmbeddingMatrix[position];
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
    const countMatrix = (matrix: number[][]): number =>
      matrix.reduce((sum, row) => sum + row.length, 0);

    return countMatrix(this.textEmbeddingMatrix) 
    + countMatrix(this.positionEmbeddingMatrix);
  }

  /**
   * Logs a one or two dimensional array. When DEBUG.VERBOSE is true the
   * whole array is printed with console.table; otherwise only its
   * dimensions are printed, which keeps the output short for large arrays.
   *
   * @param label - Name of the array, printed before its contents or dimensions.
   * @param array - The array to log: a vector, or a matrix with one row per token.
   */
  private logArray(label: string, array: number[] | number[][]): void {
    if (DEBUG.VERBOSE) {
      console.log(`Embedding| ${label}:`);
      console.table(array);
    } else if (Array.isArray(array[0])) {
      console.log(`Embedding| ${label}. Rows: ${array.length}; columns: ${array[0].length}`);
    } else {
      console.log(`Embedding| ${label}. Length: ${array.length}`);
    }
  }
}

export default Embedding;
