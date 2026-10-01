import Hyperparameters from "./Hyperparameters.js";
import MultiHeadAttention from "./MultiHeadAttention.js";
import FeedForward from "./FeedForward.js";
import LayerNormalization from "./LayerNormalization.js";
import DEBUG from "./Debug.js";

/**
 * A single GPT transformer block. It combines multi-head attention, which
 * lets each token gather information from the other tokens in the
 * sequence, with a feed forward network, which transforms each token
 * independently. Layer normalization is applied to the input of each of
 * these two stages:
 *
 * 1. Layer normalization of each token embedding.
 * 2. Multi-head attention over the normalized embeddings.
 * 3. A shortcut connection that adds the input embeddings to the attention output.
 * 4. Layer normalization of each resulting vector.
 * 5. The feed forward network, applied to each normalized vector.
 * 6. Dropout on the feed forward output (training mode only).
 * 7. A shortcut connection that adds the result of step 3 to the output of step 6.
 */
class Transformer {
  /** The multi-head attention stage of the block. */
  multiHeadAttention: MultiHeadAttention;

  /** The feed forward network stage of the block. */
  feedForward: FeedForward;

  /** Probability, between 0 and 1, that each value of the feed forward output is dropped (set to zero) by dropout. */
  dropoutRate: number;

  /** True in training mode, when dropout is applied to the feed forward output; false in inference mode, when it is skipped. */
  training: boolean;

  /**
   * Creates a Transformer with its own MultiHeadAttention and FeedForward
   * instances, both built from the same Hyperparameters so their
   * dimensions match, and copies dropoutRate and training from them.
   *
   * @param hyperparameters - Source of dropoutRate and training, and passed on to MultiHeadAttention and FeedForward. Defaults to a new Hyperparameters instance.
   */
  constructor(hyperparameters: Hyperparameters = new Hyperparameters()) {
    this.dropoutRate = hyperparameters.dropoutRate;
    this.training = hyperparameters.training;
    this.multiHeadAttention = new MultiHeadAttention(hyperparameters);
    this.feedForward = new FeedForward(hyperparameters);
  }

  /**
   * Runs the token embeddings through the transformer block: layer
   * normalization, multi-head attention, a shortcut connection adding the
   * input embeddings to the attention output, layer normalization, the
   * feed forward network, dropout (training mode only), then a second
   * shortcut connection adding the attention stage's output to the
   * feed forward output. Layer
   * normalization and the feed forward network are applied to each token
   * vector separately.
   *
   * @param embeddings - Token embeddings: one row per token, each row embeddingSize values wide.
   * @returns The output of the block: one row per token, in token order, each row embeddingSize values wide.
   */
  calculate(embeddings: number[][]): number[][] {
    // Normalize each token embedding independently before attention.
    const normalizedEmbeddings = embeddings.map((embedding) => LayerNormalization.calculate(embedding));

    const attentionOutput = this.multiHeadAttention.calculate(normalizedEmbeddings);

    /*
     * Shortcut (residual) connection: add the original, unnormalized input
     * embeddings to the attention output, element by element. This gives
     * gradients a direct path back through the block during training, and
     * means attention only has to learn a change to each embedding rather
     * than a complete replacement for it.
     */
    const shortcutOutput = attentionOutput.map((vector, token) =>
      vector.map((value, index) => value + embeddings[token][index])
    );

    // Normalize each shortcut output vector independently before the feed forward network.
    const normalizedShortcutOutput = shortcutOutput.map((vector) => LayerNormalization.calculate(vector));

    // The feed forward network processes one token vector at a time.
    const feedForwardOutput = normalizedShortcutOutput.map((vector) => this.feedForward.calculate(vector));

    /*
     * Dropout on the feed forward output, only in training mode. Each value
     * is set to 0 with probability dropoutRate; the values that are kept are
     * scaled by 1 / (1 - dropoutRate) so the expected size of each value is
     * unchanged, as in GPTAttention. In inference mode the output is used
     * unchanged.
     */
    let droppedFeedForwardOutput = feedForwardOutput;
    if (this.training) {
      const keepScale = 1 / (1 - this.dropoutRate);
      droppedFeedForwardOutput = feedForwardOutput.map((vector) =>
        vector.map((value) => (Math.random() < this.dropoutRate ? 0 : value * keepScale))
      );

      if (DEBUG.TRANSFORMER) {
        console.log(`Transformer| Feed forward output after dropout (dropout rate ${this.dropoutRate}):`);
        console.table(droppedFeedForwardOutput);
      }
    }

    /*
     * Second shortcut connection: add the output of the attention stage
     * (including the first shortcut) to the feed forward output, element by
     * element. As in GPT-2, this keeps an unbroken path from the block's
     * input to its output, so the feed forward network only has to learn a
     * change to each token vector.
     */
    const output = droppedFeedForwardOutput.map((vector, token) =>
      vector.map((value, index) => value + shortcutOutput[token][index])
    );

    if (DEBUG.TRANSFORMER) {
      console.log("Transformer| Output (one row per token):");
      console.table(output);
    }

    return output;
  }

  /**
   * Counts the trainable parameters in the block: those of the multi-head
   * attention stage plus those of the feed forward network. Layer
   * normalization, dropout and the shortcut connections have no parameters.
   *
   * @returns The total number of parameters in the block.
   */
  getParameterCount(): number {
    return this.multiHeadAttention.getParameterCount() + this.feedForward.getParameterCount();
  }
}

export default Transformer;
