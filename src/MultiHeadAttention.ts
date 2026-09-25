import Hyperparameters from "./Hyperparameters.js";
import GPTAttention from "./GPTAttention.js";
import DEBUG from "./Debug.js";

/**
 * Runs several GPTAttention heads over the same token embeddings and
 * concatenates their context vectors. Each head has its own independently
 * initialized query, key and value weight matrices, so each can learn to
 * attend to different relationships between tokens. The number of heads is
 * taken from Hyperparameters.numberAttentionHeads.
 */
class MultiHeadAttention {
  /** The attention heads, one GPTAttention instance per head, in head order. */
  attentionHeads: GPTAttention[];

  /**
   * Creates a MultiHeadAttention with numberAttentionHeads GPTAttention
   * heads, each built from the same Hyperparameters so they all share the
   * same dimensions, dropout rate and training mode.
   *
   * @param hyperparameters - Source of numberAttentionHeads, and passed on to each GPTAttention head. Defaults to a new Hyperparameters instance.
   */
  constructor(hyperparameters: Hyperparameters = new Hyperparameters()) {
    this.attentionHeads = Array.from(
      { length: hyperparameters.numberAttentionHeads },
      () => new GPTAttention(hyperparameters)
    );

    if(DEBUG.ATTENTION) console.log(`MultiHeadAttention| Created ${this.attentionHeads.length} attention heads.`);
  }

  /**
   * Calls calculate() on each attention head in turn with the same
   * embeddings, then concatenates the heads' context vectors token by
   * token. Row i of the result is head 0's context vector for token i,
   * followed by head 1's, and so on, giving a tokenCount x
   * (numberAttentionHeads * weightMatrixColumns) matrix.
   *
   * @param embeddings - Token embeddings as produced by Embedding.getEmbedding(): one row per token, each row embeddingSize values wide.
   * @returns The concatenated context vectors: one row per token, in token order, each holding every head's context vector for that token, in head order.
   */
  calculate(embeddings: number[][]): number[][] {
    // One tokenCount x weightMatrixColumns matrix of context vectors per head.
    const headContextVectors: number[][][] = this.attentionHeads.map((head, headIndex) => {
      if(DEBUG.ATTENTION) console.log(`MultiHeadAttention| Calculating attention head ${headIndex}.`);
      return head.calculate(embeddings);
    });

    /*
     * Concatenate along the last dimension: for each token, join the
     * context vectors that every head produced for that token into a
     * single row, in head order.
     */
    const contextVectors: number[][] = embeddings.map((_, token) =>
      headContextVectors.flatMap((headVectors) => headVectors[token])
    );

    if (DEBUG.CONTEXT) {
      console.log("MultiHeadAttention| Concatenated context vectors (one row per token):");
      console.table(contextVectors);
    }

    return contextVectors;
  }
}

export default MultiHeadAttention;
