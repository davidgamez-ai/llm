import Hyperparameters from "./Hyperparameters.js";
import DEBUG from "./Debug.js";

/**
 * The linear output layer of the GPT model. It holds a weight matrix with
 * one row per vocabulary token, each row embeddingSize values wide, which is
 * used to map each token vector output by the model to a score for every
 * token in the vocabulary. The vocabulary size and embedding size are taken
 * from Hyperparameters, so they always match the values used elsewhere in
 * the application.
 */
class LinearOutputLayer {
  /** Number of rows in the weight matrix, i.e. the number of tokens in the vocabulary. */
  vocabSize: number;

  /** Number of columns in the weight matrix, i.e. the width of each token vector. */
  embeddingSize: number;

  /** True in training mode, when logits are calculated for every token; false in inference mode, when they are only calculated for the last token. */
  training: boolean;

  /** The weight matrix built by build(): vocabSize rows by embeddingSize columns. */
  weights: number[][] = [];

  /** Draws from the Hyperparameters' shared random number stream, which is seeded when Hyperparameters.seed is set. */
  private readonly random: () => number;

  /**
   * Creates a LinearOutputLayer, copying vocabSize, embeddingSize and
   * training from the given Hyperparameters so this instance always matches
   * the values used elsewhere in the application.
   *
   * @param hyperparameters - Source of vocabularySize, embeddingSize, training and the random number stream. Defaults to a new Hyperparameters instance.
   */
  constructor(hyperparameters: Hyperparameters = new Hyperparameters()) {
    this.vocabSize = hyperparameters.vocabularySize;
    this.embeddingSize = hyperparameters.embeddingSize;
    this.training = hyperparameters.training;
    this.random = () => hyperparameters.random();
  }

  /**
   * Builds the vocabSize x embeddingSize weight matrix, filled with values
   * drawn uniformly from [min, max), and stores it on the instance. Uses the
   * same initialization as Embedding.build. Random values come from the
   * Hyperparameters' shared stream, so the matrix is reproducible when
   * Hyperparameters.seed is set.
   *
   * @param min - Inclusive lower bound of the random range. Defaults to -3.
   * @param max - Exclusive upper bound of the random range. Defaults to 3.
   * @returns void. The resulting matrix is stored on the instance.
   */
  build(min: number = -3, max: number = 3): void {
    this.weights = Array.from({ length: this.vocabSize }, () =>
      Array.from({ length: this.embeddingSize }, () => min + this.random() * (max - min))
    );

    if (DEBUG.LINEAR_OUTPUT_LAYER) console.log(
      `LinearOutputLayer| Weight matrix built. Rows: ${this.weights.length}; columns: ${this.weights[0].length}`
    );
  }

  /**
   * Maps each token embedding to a score (logit) for every token in the
   * vocabulary. The score for vocabulary token v is the dot product of the
   * embedding with row v of the weight matrix, so the whole calculation is
   * embeddings (numberOfTokens x embeddingSize) multiplied by the transpose
   * of weights (embeddingSize x vocabSize).
   *
   * In training mode logits are calculated for every token, since the model
   * learns to predict the next token at every position in the sequence. In
   * inference mode only the last token's logits are needed to predict the
   * next token, so the other tokens are skipped, and a softmax is applied to
   * turn the logits into probabilities.
   *
   * @param embeddings - Token embeddings: one row per token, each row embeddingSize values wide.
   * @returns Each row vocabSize values wide: in training mode the logits, one row per token, in token order; in inference mode a single row of softmax probabilities for the last token.
   */
  calculate(embeddings: number[][]): number[][] {
    const selectedEmbeddings = this.training ? embeddings : embeddings.slice(-1);

    const logits = selectedEmbeddings.map((embedding) =>
      this.weights.map((weightRow) =>
        weightRow.reduce((sum, weight, index) => sum + weight * embedding[index], 0)
      )
    );

    /*
     * In inference mode, apply a softmax to each row of logits to turn them
     * into a probability distribution over the vocabulary: every value
     * positive and the row summing to 1. The row's maximum is subtracted
     * before exponentiating to avoid overflow; this does not change the
     * result because softmax is unaffected by adding a constant to every
     * input. The maximum is found with reduce rather than Math.max(...row),
     * because spreading a vocabulary-sized row as arguments can exceed the
     * engine's argument limit.
     */
    const output = this.training
      ? logits
      : logits.map((row) => {
          const maxLogit = row.reduce((max, value) => Math.max(max, value), -Infinity);
          const exponentiated = row.map((value) => Math.exp(value - maxLogit));
          const total = exponentiated.reduce((sum, value) => sum + value, 0);
          return exponentiated.map((value) => value / total);
        });

    // Print the whole output in verbose mode; otherwise just its dimensions.
    if (DEBUG.LINEAR_OUTPUT_LAYER) {
      if (DEBUG.VERBOSE) {
        console.log("LinearOutputLayer| Output (one row per token):");
        console.table(output);
      } else {
        console.log(`LinearOutputLayer| Output. Rows: ${output.length}; columns: ${output[0]?.length ?? 0}`);
      }
    }

    return output;
  }

  /**
   * Counts the trainable parameters in the layer: every value in the weight
   * matrix. Returns 0 until build() has been called.
   *
   * @returns The total number of weights currently stored in weights.
   */
  getParameterCount(): number {
    return this.weights.reduce((sum, row) => sum + row.length, 0);
  }
}

export default LinearOutputLayer;
