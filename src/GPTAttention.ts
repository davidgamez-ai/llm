import Hyperparameters from "./Hyperparameters.js";
import { randomNormalMatrix } from "./WeightInitialization.js";
import Matrix, { vectorTimesMatrix } from "./Matrix.js";
import DEBUG from "./Debug.js"

/**
 * Holds the trainable query, key and value weight matrices used by
 * GPT-style self-attention to project each token embedding into its
 * query, key and value vectors. Each matrix is embeddingSize rows by
 * weightMatrixColumns columns, both taken from Hyperparameters, so they
 * always match the values used elsewhere in the application.
 */
class GPTAttention {
  /** Number of rows in each weight matrix, i.e. the width of each input embedding vector. */
  embeddingSize: number;

  /** Number of columns in each weight matrix, i.e. the width of each projected query/key/value vector. */
  weightMatrixColumns: number;

  /** Probability, between 0 and 1, that each attention weight is dropped (set to zero) by dropout. */
  dropoutRate: number;

  /**
   * The current mode: Hyperparameters.TRAINING_MODE, VALIDATION_MODE or
   * GENERATION_MODE. Dropout is only applied to the attention weights in
   * TRAINING_MODE. Read from Hyperparameters on every access, so changing
   * Hyperparameters.mode after construction takes effect.
   */
  get mode(): number {
    return this.getMode();
  }

  /** Reads the Hyperparameters' current mode. */
  private readonly getMode: () => number;

  /** The query weight matrix built by build(): embeddingSize rows by weightMatrixColumns columns. */
  queryWeights: Matrix = new Matrix(0, 0);

  /** The key weight matrix built by build(): embeddingSize rows by weightMatrixColumns columns. */
  keyWeights: Matrix = new Matrix(0, 0);

  /** The value weight matrix built by build(): embeddingSize rows by weightMatrixColumns columns. */
  valueWeights: Matrix = new Matrix(0, 0);

  /** True if the query, key and value projections add a bias to their outputs; false if they do not. */
  qkvBias: boolean;

  /** Query biases built by build(): weightMatrixColumns values. Empty when qkvBias is false. */
  queryBiases: Float32Array = new Float32Array(0);

  /** Key biases built by build(): weightMatrixColumns values. Empty when qkvBias is false. */
  keyBiases: Float32Array = new Float32Array(0);

  /** Value biases built by build(): weightMatrixColumns values. Empty when qkvBias is false. */
  valueBiases: Float32Array = new Float32Array(0);

  /** Standard deviation of the normal distribution (mean 0) used to initialize the weight matrices. GPT-2 uses 0.02. */
  standardDeviation: number;

  /** Draws from the Hyperparameters' shared random number stream, which is seeded when Hyperparameters.seed is set. */
  private readonly random: () => number;

  /**
   * Creates a GPTAttention, copying embeddingSize, weightMatrixColumns,
   * dropoutRate, qkvBias and linearLayerStandardDeviation from the given
   * Hyperparameters so this instance always matches the values used
   * elsewhere in the application, then builds the weight matrices and biases.
   *
   * @param hyperparameters - Source of embeddingSize, weightMatrixColumns, dropoutRate, mode, qkvBias, linearLayerStandardDeviation and the random number stream. Defaults to a new Hyperparameters instance.
   */
  constructor(hyperparameters: Hyperparameters = new Hyperparameters()) {
    this.embeddingSize = hyperparameters.embeddingSize;
    this.weightMatrixColumns = hyperparameters.weightMatrixColumns;
    this.dropoutRate = hyperparameters.dropoutRate;
    this.getMode = () => hyperparameters.mode;
    this.qkvBias = hyperparameters.qkvBias;
    this.standardDeviation = hyperparameters.linearLayerStandardDeviation;
    this.random = () => hyperparameters.random();

    this.build();
  }

  /**
   * Builds the queryWeights, keyWeights and valueWeights matrices, each
   * embeddingSize rows by weightMatrixColumns columns, and stores them on the
   * instance. Initialized as in GPT-2 (Hugging Face
   * GPT2PreTrainedModel._init_weights, which initializes GPT-2's combined
   * query/key/value layer c_attn this way):
   * - Weights are drawn from a normal distribution with mean 0 and standard
   *   deviation standardDeviation (0.02 by default).
   * - When qkvBias is true, each projection has a bias vector of
   *   weightMatrixColumns values, all 0; otherwise the bias arrays are left
   *   empty.
   *
   * @returns void. The resulting matrices and biases are stored on the instance.
   */
  build(): void {
    // Called once per weight matrix, so each gets its own independent set of random values.
    this.queryWeights = randomNormalMatrix(this.embeddingSize, this.weightMatrixColumns, this.standardDeviation, this.random);
    this.keyWeights = randomNormalMatrix(this.embeddingSize, this.weightMatrixColumns, this.standardDeviation, this.random);
    this.valueWeights = randomNormalMatrix(this.embeddingSize, this.weightMatrixColumns, this.standardDeviation, this.random);

    // A new Float32Array is filled with zeros, so the biases start at 0.
    const biasLength: number = this.qkvBias ? this.weightMatrixColumns : 0;
    this.queryBiases = new Float32Array(biasLength);
    this.keyBiases = new Float32Array(biasLength);
    this.valueBiases = new Float32Array(biasLength);

    if(DEBUG.QKV) console.log(
      `GPTAttention| Weight matrices built. Rows: ${this.embeddingSize}; columns: ${this.weightMatrixColumns}`
    );

    // console.table lays each matrix out as a proper grid of rows and
    // columns, which is far easier to read than the flattened,
    // comma-separated string that logging a nested array directly produces.
    if(DEBUG.QKV) {
      console.log("GPTAttention| Query weights:");
      console.table(this.queryWeights.toArrays());
      console.log("GPTAttention| Key weights:");
      console.table(this.keyWeights.toArrays());
      console.log("GPTAttention| Value weights:");
      console.table(this.valueWeights.toArrays());
    }
  }

  /**
   * For each token's embedding vector, separately multiplies it by
   * queryWeights, keyWeights and valueWeights to project it into that
   * token's query, key and value vectors. Collecting these vectors across
   * every token produces three new matrices: queries, keys and values,
   * each tokenCount rows by weightMatrixColumns columns. Each token's query
   * is then multiplied (dot product) by every token's key to give that
   * token's attention scores. These scores are scaled by the square root of
   * the key dimension, and a causal mask sets the scores above the diagonal
   * (tokens later in the sequence) to -Infinity. A softmax across each row
   * then gives attention weights in which the masked tokens are 0 and each
   * row sums to 1. In training mode dropout is then applied
   * to give the final attention weights; in validation and generation modes
   * it is skipped.
   * Finally, each token's context vector is the sum of every token's value
   * vector multiplied by the corresponding final attention weight. The
   * intermediate matrices are logged but not returned.
   *
   * @param embeddings - Token embeddings as produced by Embedding.getEmbedding(): one row per token, each row embeddingSize values wide.
   * @returns The context vectors: a tokenCount x weightMatrixColumns matrix where row i is the causally masked, attention-weighted (and, in training mode, dropout-applied) sum of the value vectors of tokens 0..i.
   */
  calculate(embeddings: number[][]): number[][] {
    if(DEBUG.ATTENTION) console.log(
      `GPTAttention| Calculating queries, keys and values for ${embeddings.length} tokens.`
    );

    const queries: number[][] = [];
    const keys: number[][] = [];
    const values: number[][] = [];

    /*
     * For each token in turn, project its embedding vector into its query,
     * key and value vectors by multiplying it by the three weight matrices
     * and adding the biases (which are empty when qkvBias is false). A
     * 1 x embeddingSize vector times an embeddingSize x weightMatrixColumns
     * matrix produces a 1 x weightMatrixColumns vector.
     */
    for (let i = 0; i < embeddings.length; i++) {
      const embedding = embeddings[i];

      const query = vectorTimesMatrix(embedding, this.queryWeights, this.queryBiases);
      const key = vectorTimesMatrix(embedding, this.keyWeights, this.keyBiases);
      const value = vectorTimesMatrix(embedding, this.valueWeights, this.valueBiases);

      queries.push(query);
      keys.push(key);
      values.push(value);
    }

    // console.table lays each resulting matrix out as a proper grid of
    // rows (tokens) and columns, which is far easier to check by eye than
    // logging a nested array directly, row by row, as the loop went along.
    if(DEBUG.ATTENTION) {
      console.log("GPTAttention| Queries (one row per token):");
      console.table(queries);
      console.log("GPTAttention| Keys (one row per token):");
      console.table(keys);
      console.log("GPTAttention| Values (one row per token):");
      console.table(values);
    }

    const tokenCount: number = embeddings.length;

    /*
     * For each token, take the dot product of its query vector with the key
     * vector of every token in the sequence (including itself). The result
     * is a tokenCount x tokenCount matrix in which row i holds the attention
     * scores of token i, i.e. how strongly token i's query matches each
     * token's key:
     *
     *     score[i][j] = sum over d of query[i][d] * key[j][d]
     */
    const attentionScores: number[][] = [];
    for (let i = 0; i < tokenCount; i++) {
      attentionScores.push([]);
      for (let j = 0; j < tokenCount; j++) {
        let dotProduct = 0;
        for (let d = 0; d < this.weightMatrixColumns; d++) {
          dotProduct += queries[i][d] * keys[j][d];
        }
        attentionScores[i].push(dotProduct);
      }
    }

    if(DEBUG.ATTENTION) {
      console.log("GPTAttention| Attention scores (row = query token, column = key token):");
      console.table(attentionScores);
    }

    /*
     * Scale each token's attention scores by dividing them by the square
     * root of the key dimension (weightMatrixColumns). Scaling stops the dot
     * products growing with the key dimension, which would otherwise push
     * the softmax towards a one-hot output with tiny gradients.
     *
     * At the same time apply the causal mask: the score of every token that
     * comes after token i in the sequence (column > row, above the diagonal)
     * is replaced with -Infinity, so token i can only attend to tokens 0..i.
     * Masking before the softmax, as the book does, means the masked tokens
     * take no part in the softmax at all, so no re-normalization is needed
     * afterwards.
     */
    const scale = Math.sqrt(this.weightMatrixColumns);
    const maskedScores: number[][] = [];
    for (let i = 0; i < tokenCount; i++) {
      maskedScores.push([]);
      for (let j = 0; j < tokenCount; j++) {
        // Token i may attend to token j only if j is not after i.
        maskedScores[i].push(j <= i ? attentionScores[i][j] / scale : -Infinity);
      }
    }

    if(DEBUG.ATTENTION) {
      console.log("GPTAttention| Scaled and causally masked attention scores (-Infinity = masked):");
      console.table(maskedScores);
    }

    /*
     * Apply a softmax across each row to turn the masked scores into
     * attention weights that are all positive and sum to 1. Because
     * exp(-Infinity) is 0, the masked tokens get a weight of exactly 0.
     *
     * The row's maximum is subtracted before exponentiating to avoid
     * overflow; this does not change the result because softmax is
     * unaffected by adding a constant to every input. The maximum is taken
     * after masking, so it always comes from a visible token. The visible
     * token with that maximum therefore has exp(0) = 1, which means the row
     * total is at least 1 and can never underflow to 0. (Taking the maximum
     * before masking could make every visible exponential underflow to 0 if
     * a future token's score was much larger, giving a NaN row.) The diagonal
     * is never masked, so every row has at least one finite score.
     */
    const attentionWeights: number[][] = maskedScores.map((row) => {
      const max = Math.max(...row);
      const exponentials = row.map((score) => Math.exp(score - max));
      const total = exponentials.reduce((sum, value) => sum + value, 0);
      return exponentials.map((value) => value / total);
    });

    if(DEBUG.ATTENTION) {
      console.log("GPTAttention| Attention weights (masked softmax; values above the diagonal are 0, each row sums to 1):");
      console.table(attentionWeights);
    }

    /*
     * Dropout is only applied in training mode. In validation and
     * generation modes the attention weights are used unchanged as the
     * final attention weights, so the output is deterministic.
     */
    let finalAttentionWeights: number[][];
    if (this.mode === Hyperparameters.TRAINING_MODE) {
      /*
       * Build a dropout mask with the same tokenCount x tokenCount
       * dimensions as the attention weights. Each entry is independently
       * set to 0 with probability dropoutRate, so on average that
       * proportion of the mask is zero. The remaining entries are set to
       * 1 / (1 - dropoutRate) so that, once the mask is applied, the
       * expected value of each attention weight is unchanged. With the
       * default dropoutRate of 0.1 the kept entries are 1 / 0.9, about 1.11.
       */
      const keepScale = 1 / (1 - this.dropoutRate);
      const dropoutMask: number[][] = attentionWeights.map((row) =>
        row.map(() => (this.random() < this.dropoutRate ? 0 : keepScale))
      );
      if(DEBUG.ATTENTION) {
        console.log(`GPTAttention| Dropout mask (dropout rate ${this.dropoutRate}; kept values = ${keepScale}):`);
        console.table(dropoutMask);
      }

      /*
       * Apply dropout by multiplying the attention weights element by
       * element with the dropout mask. Dropped weights become 0 and kept
       * weights are scaled by 1 / (1 - dropoutRate).
       */
      finalAttentionWeights = [];
      for (let i = 0; i < tokenCount; i++) {
        finalAttentionWeights.push([]);
        for (let j = 0; j < tokenCount; j++) {
          finalAttentionWeights[i].push(attentionWeights[i][j] * dropoutMask[i][j]);
        }
      }

      if(DEBUG.ATTENTION) console.log("GPTAttention| Final attention weights (after causal mask and dropout):");
    } else {
      finalAttentionWeights = attentionWeights;

      if(DEBUG.ATTENTION) console.log("GPTAttention| Validation or generation mode: dropout skipped. Final attention weights (after causal mask):");
    }
    if(DEBUG.ATTENTION) console.table(finalAttentionWeights);

    /*
     * For each token, multiply the value vector of every token (including
     * itself) by the corresponding weight from that token's row of the
     * final attention weights, then sum the weighted value vectors. The
     * result is the token's context vector: a blend of the value vectors of
     * that token and the tokens before it, in which the tokens it attends to
     * most contribute most. Later tokens and dropped weights are zero, so
     * they do not contribute. Collecting these gives a tokenCount x
     * weightMatrixColumns matrix:
     *
     *     context[i][d] = sum over j of weight[i][j] * value[j][d]
     */
    const contextVectors: number[][] = [];
    for (let i = 0; i < tokenCount; i++) {
      contextVectors.push([]);
      for (let d = 0; d < this.weightMatrixColumns; d++) {
        let sum = 0;
        for (let j = 0; j < tokenCount; j++) {
          sum += finalAttentionWeights[i][j] * values[j][d];
        }
        contextVectors[i].push(sum);
      }
    }

    if(DEBUG.CONTEXT) {
      console.log("GPTAttention| Context vectors (one row per token):");
      console.table(contextVectors);
    }

    return contextVectors;
  }

  /**
   * Counts the trainable parameters in this attention head: every value in
   * the query, key and value weight matrices, plus their biases. The biases
   * are empty, and so add nothing, when qkvBias is false.
   *
   * @returns The total number of weights and biases currently stored on the instance.
   */
  getParameterCount(): number {
    return (
      this.queryWeights.size + this.keyWeights.size + this.valueWeights.size +
      this.queryBiases.length + this.keyBiases.length + this.valueBiases.length
    );
  }
}

export default GPTAttention;
