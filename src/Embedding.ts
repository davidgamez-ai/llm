import BPETokenizer from "./BPETokenizer.js";
import Hyperparameters from "./Hyperparameters.js";

/**
 * Two embedding matrices, each mapping an id (a token id, or a sequence
 * position) to a vector of embeddingSize random values. The vocabulary size
 * and embedding size are taken from Hyperparameters, so they always match
 * the values used elsewhere in the application.
 */
class Embedding {
  /** Number of rows in each embedding matrix, i.e. the number of distinct token ids supported. */
  vocabSize: number;

  /** Number of columns in each embedding matrix, i.e. the width of each embedding vector. */
  embeddingSize: number;

  // The token embedding matrix built by build(): one row per vocabulary
  // token, each row embeddingSize values wide. Looked up by token id.
  private textEmbeddingMatrix: number[][] = [];

  // The position embedding matrix built by build(): same dimensions as
  // textEmbeddingMatrix, but looked up by a token's position in the
  // sequence rather than by its id.
  private positionEmbeddingMatrix: number[][] = [];

  /**
   * Creates an Embedding, copying vocabSize and embeddingSize from the given
   * Hyperparameters so this instance always matches the values used
   * elsewhere in the application.
   *
   * @param hyperparameters - Source of vocabularySize and embeddingSize. Defaults to a new Hyperparameters instance.
   */
  constructor(hyperparameters: Hyperparameters = new Hyperparameters()) {
    this.vocabSize = hyperparameters.vocabularySize;
    this.embeddingSize = hyperparameters.embeddingSize;
  }

  /**
   * Builds two vocabSize x embeddingSize matrices, textEmbeddingMatrix and
   * positionEmbeddingMatrix, each filled with values drawn uniformly from
   * [min, max), stores them on the instance, and logs their dimensions.
   * Both matrices share the same shape and random-generation procedure;
   * they differ only in how getEmbedding looks rows up in them (by token id
   * versus by sequence position).
   *
   * @param seed - Optional PRNG seed. When provided, the same seed always produces the same matrices, since Math.random() cannot be seeded and would otherwise make results unreproducible between runs. When omitted, Math.random() is used and the matrices differ on every call.
   * @param min - Inclusive lower bound of the random range. Defaults to -5.
   * @param max - Exclusive upper bound of the random range. Defaults to 5.
   * @returns void. The resulting matrices are stored on the instance for use by getEmbedding.
   */
  build(seed?: number, min: number = -3, max: number = 3): void {
    const random = seed === undefined ? Math.random : this.createSeededRandom(seed);

    const buildMatrix = () =>
      Array.from({ length: this.vocabSize }, () =>
        Array.from({ length: this.embeddingSize }, () => min + random() * (max - min))
      );

    this.textEmbeddingMatrix = buildMatrix();
    this.positionEmbeddingMatrix = buildMatrix();

    console.log(
      `Embeddings| Embedding matrices built. Rows: ${this.textEmbeddingMatrix.length}; columns: ${this.textEmbeddingMatrix[0].length}`
    );
  }

  /**
   * Tokenizes text with BPETokenizer, then for each resulting token looks up
   * its row in textEmbeddingMatrix by token id, and its row in
   * positionEmbeddingMatrix by the token's position in the sequence, and
   * adds the two rows together. Each stage (token ids, token embeddings,
   * position embeddings, combined embeddings) is stored in its own array and
   * logged so intermediate results can be inspected, rather than only
   * returning the final combined result.
   *
   * @param text - Raw input text to embed.
   * @returns One combined embedding vector per token, in token order. Each vector is the token's textEmbeddingMatrix row plus its positionEmbeddingMatrix row.
   */
  getEmbedding(text: string): number[][] {
    const tokenIds = new BPETokenizer().encode(text);

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
    // console.log("Embedding| Token ids:", tokenIds);
    // console.log("Embedding| Token embeddings:", tokenEmbeddings);
    // console.log("Embedding| Position embeddings:", positionEmbeddings);
    // console.log("Embedding| Combined embeddings:", combinedEmbeddings);

    return combinedEmbeddings;
  }

  /**
   * mulberry32: a small, fast seeded PRNG. Deterministically derives a
   * stream of 32-bit states from the seed and scrambles each one via
   * Math.imul/xorshift into a uniform value in [0, 1), so build() can
   * produce a reproducible embedding matrix for a given seed.
   *
   * @param seed - Initial 32-bit PRNG state.
   * @returns A function that, on each call, advances the PRNG state and returns the next pseudo-random value in [0, 1).
   */
  private createSeededRandom(seed: number): () => number {
    let state = seed;

    return () => {
      // Advance the state with a fixed odd increment (32-bit overflow wraps via `| 0`).
      state = (state + 0x6d2b79f5) | 0;

      // Scramble the state (xorshift + multiply) so consecutive states don't correlate.
      let t = Math.imul(state ^ (state >>> 15), 1 | state);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;

      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
}

export default Embedding;
