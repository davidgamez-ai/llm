import Hyperparameters from "./Hyperparameters.js";

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

  /** The query weight matrix built by build(): embeddingSize rows by weightMatrixColumns columns. */
  queryWeights: number[][] = [];

  /** The key weight matrix built by build(): embeddingSize rows by weightMatrixColumns columns. */
  keyWeights: number[][] = [];

  /** The value weight matrix built by build(): embeddingSize rows by weightMatrixColumns columns. */
  valueWeights: number[][] = [];

  /**
   * Creates a GPTAttention, copying embeddingSize and weightMatrixColumns
   * from the given Hyperparameters so this instance always matches the
   * values used elsewhere in the application, then builds the weight
   * matrices.
   *
   * @param hyperparameters - Source of embeddingSize and weightMatrixColumns. Defaults to a new Hyperparameters instance.
   */
  constructor(hyperparameters: Hyperparameters = new Hyperparameters()) {
    this.embeddingSize = hyperparameters.embeddingSize;
    this.weightMatrixColumns = hyperparameters.weightMatrixColumns;

    this.build();
  }

  /**
   * Builds the queryWeights, keyWeights and valueWeights matrices, each
   * embeddingSize rows by weightMatrixColumns columns, and stores them on the
   * instance. Weights are initialized the same way as PyTorch's nn.Linear:
   * drawn uniformly from [-1/sqrt(embeddingSize), 1/sqrt(embeddingSize)).
   *
   * @returns void. The resulting matrices are stored on the instance.
   */
  build(): void {
    /*
     * PyTorch's nn.Linear uses kaiming_uniform_ with a = sqrt(5), which
     * works out as a uniform distribution over [-bound, bound) with
     * bound = 1 / sqrt(fan_in). Here fan_in is the number of inputs to each
     * output, i.e. embeddingSize. Scaling the range by fan_in keeps the
     * variance of the projected query, key and value vectors roughly
     * independent of the embedding size.
     */
    const bound = 1 / Math.sqrt(this.embeddingSize);

    // A fresh embeddingSize x weightMatrixColumns matrix of random values
    // in [-bound, bound); called once per weight matrix so each gets its own
    // independent set of random values.
    const buildMatrix = () =>
      Array.from({ length: this.embeddingSize }, () =>
        Array.from({ length: this.weightMatrixColumns }, () => (Math.random() * 2 - 1) * bound)
      );

    this.queryWeights = buildMatrix();
    this.keyWeights = buildMatrix();
    this.valueWeights = buildMatrix();

    console.log(
      `GPTAttention| Weight matrices built. Rows: ${this.embeddingSize}; columns: ${this.weightMatrixColumns}`
    );

    // console.table lays each matrix out as a proper grid of rows and
    // columns, which is far easier to read than the flattened,
    // comma-separated string that logging a nested array directly produces.
    console.log("GPTAttention| Query weights:");
    console.table(this.queryWeights);
    console.log("GPTAttention| Key weights:");
    console.table(this.keyWeights);
    console.log("GPTAttention| Value weights:");
    console.table(this.valueWeights);
  }

  /**
   * For each token's embedding vector, separately multiplies it by
   * queryWeights, keyWeights and valueWeights to project it into that
   * token's query, key and value vectors. Collecting these vectors across
   * every token produces three new matrices: queries, keys and values,
   * each tokenCount rows by weightMatrixColumns columns. Each token's query
   * is then multiplied (dot product) by every token's key to give that
   * token's attention scores. These scores are scaled by the square root of
   * the key dimension and passed through a softmax to give attention weights.
   * Finally, each token's context vector is the sum of every token's value
   * vector multiplied by the corresponding attention weight.
   *
   * @param embeddings - Token embeddings as produced by Embedding.getEmbedding(): one row per token, each row embeddingSize values wide.
   * @returns An object with attentionScores (a tokenCount x tokenCount matrix where row i, column j is the dot product of token i's query with token j's key), attentionWeights (the same scores divided by the square root of the key dimension and softmaxed so that each row sums to 1) and contextVectors (a tokenCount x weightMatrixColumns matrix where row i is the attention-weighted sum of all value vectors for token i).
   */
  calculate(embeddings: number[][]): {
    attentionScores: number[][];
    attentionWeights: number[][];
    contextVectors: number[][];
  } {
    console.log(
      `GPTAttention| Calculating queries, keys and values for ${embeddings.length} tokens.`
    );

    // Multiplies a single embedding vector by a weight matrix: each column
    // of the result is the dot product of the vector with that column of
    // the matrix, so a 1 x embeddingSize vector times an embeddingSize x
    // weightMatrixColumns matrix produces a 1 x weightMatrixColumns vector.
    const multiplyVectorByMatrix = (vector: number[], matrix: number[][]): number[] =>
      Array.from({ length: this.weightMatrixColumns }, (_, column) =>
        vector.reduce((sum, value, row) => sum + value * matrix[row][column], 0)
      );

    const queries: number[][] = [];
    const keys: number[][] = [];
    const values: number[][] = [];

    // For each token in turn, project its embedding vector into its query,
    // key and value vectors using the three weight matrices.
    for (let i = 0; i < embeddings.length; i++) {
      const embedding = embeddings[i];

      const query = multiplyVectorByMatrix(embedding, this.queryWeights);
      const key = multiplyVectorByMatrix(embedding, this.keyWeights);
      const value = multiplyVectorByMatrix(embedding, this.valueWeights);

      queries.push(query);
      keys.push(key);
      values.push(value);
    }

    // console.table lays each resulting matrix out as a proper grid of
    // rows (tokens) and columns, which is far easier to check by eye than
    // logging a nested array directly, row by row, as the loop went along.
    console.log("GPTAttention| Queries (one row per token):");
    console.table(queries);
    console.log("GPTAttention| Keys (one row per token):");
    console.table(keys);
    console.log("GPTAttention| Values (one row per token):");
    console.table(values);

    /*
     * For each token, take the dot product of its query vector with the key
     * vector of every token in the sequence (including itself). The result
     * is a tokenCount x tokenCount matrix in which row i holds the attention
     * scores of token i, i.e. how strongly token i's query matches each
     * token's key.
     */
    const attentionScores: number[][] = queries.map((query) =>
      keys.map((key) => query.reduce((sum, value, index) => sum + value * key[index], 0))
    );

    console.log("GPTAttention| Attention scores (row = query token, column = key token):");
    console.table(attentionScores);

    /*
     * Scale each token's attention scores by dividing them by the square
     * root of the key dimension (weightMatrixColumns), then apply a softmax
     * across each row. Scaling stops the dot products growing with the key
     * dimension, which would otherwise push the softmax towards a one-hot
     * output with tiny gradients. The softmax turns each row into attention
     * weights that are all positive and sum to 1. The row's maximum is
     * subtracted before exponentiating to avoid overflow; this does not
     * change the result because softmax is unaffected by adding a constant
     * to every input.
     */
    const scale = Math.sqrt(this.weightMatrixColumns);
    const attentionWeights: number[][] = attentionScores.map((row) => {
      const scaled = row.map((score) => score / scale);
      const max = Math.max(...scaled);
      const exponentials = scaled.map((score) => Math.exp(score - max));
      const total = exponentials.reduce((sum, value) => sum + value, 0);
      return exponentials.map((value) => value / total);
    });

    console.log("GPTAttention| Attention weights (scaled and softmaxed; each row sums to 1):");
    console.table(attentionWeights);

    /*
     * For each token, multiply the value vector of every token (including
     * itself) by the corresponding attention weight from that token's row,
     * then sum the weighted value vectors. The result is the token's context
     * vector: a blend of all value vectors in which the tokens it attends to
     * most contribute most. Collecting these gives a tokenCount x
     * weightMatrixColumns matrix.
     */
    const contextVectors: number[][] = attentionWeights.map((weights) =>
      Array.from({ length: this.weightMatrixColumns }, (_, column) =>
        weights.reduce((sum, weight, token) => sum + weight * values[token][column], 0)
      )
    );

    console.log("GPTAttention| Context vectors (one row per token):");
    console.table(contextVectors);

    return { attentionScores, attentionWeights, contextVectors };
  }
}

export default GPTAttention;
