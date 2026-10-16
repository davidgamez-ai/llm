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

  /**
   * The current mode: Hyperparameters.TRAINING_MODE, VALIDATION_MODE or
   * GENERATION_MODE. Logits are calculated for every token in training and
   * validation modes, and only for the last token in generation mode. Read
   * from Hyperparameters on every access, so changing Hyperparameters.mode
   * after construction takes effect.
   */
  get mode(): number {
    return this.getMode();
  }

  /** Reads the Hyperparameters' current mode. */
  private readonly getMode: () => number;

  /** Standard deviation of the normal distribution (mean 0) used to initialize the weight matrix. GPT-2 uses 0.02. */
  standardDeviation: number;

  /** The weight matrix built by build(): vocabSize rows by embeddingSize columns. */
  weights: number[][] = [];

  /** Draws from the Hyperparameters' shared random number stream, which is seeded when Hyperparameters.seed is set. */
  private readonly random: () => number;

  /**
   * Creates a LinearOutputLayer, copying vocabSize, embeddingSize and
   * outputLayerStandardDeviation from the given Hyperparameters so this
   * instance always matches the values used elsewhere in the application.
   *
   * @param hyperparameters - Source of vocabularySize, embeddingSize, outputLayerStandardDeviation, mode and the random number stream. Defaults to a new Hyperparameters instance.
   */
  constructor(hyperparameters: Hyperparameters = new Hyperparameters()) {
    this.vocabSize = hyperparameters.vocabularySize;
    this.embeddingSize = hyperparameters.embeddingSize;
    this.standardDeviation = hyperparameters.outputLayerStandardDeviation;
    this.getMode = () => hyperparameters.mode;
    this.random = () => hyperparameters.random();
  }

  /**
   * Builds the vocabSize x embeddingSize weight matrix and stores it on the
   * instance. Initialized as in GPT-2 (Hugging Face GPT2Model._init_weights),
   * with values drawn from a normal distribution with mean 0 and standard
   * deviation standardDeviation (0.02 by default). Random values come from
   * the Hyperparameters' shared stream, so the matrix is reproducible when
   * Hyperparameters.seed is set.
   *
   * @returns void. The resulting matrix is stored on the instance.
   */
  build(): void {
    /*
     * Draws one sample from a normal distribution with mean 0 and the given
     * standard deviation, using the Box-Muller transform to turn two uniform
     * samples into a standard normal sample. 1 - random() lies in (0, 1],
     * so Math.log never receives 0.
     */
    const randomNormal = (deviation: number): number => {
      const u1 = 1 - this.random();
      const u2 = this.random();
      return deviation * Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    };

    /*
     * A small standard deviation keeps the initial logits close together:
     * each logit is a dot product of embeddingSize terms with a normalized
     * vector, so its standard deviation is about
     * standardDeviation * sqrt(embeddingSize), roughly 0.55 for 0.02 and 768.
     * The untrained model therefore gives every token a similar probability,
     * and the initial loss is close to ln(vocabSize), about 10.8.
     */
    this.weights = Array.from({ length: this.vocabSize }, () =>
      Array.from({ length: this.embeddingSize }, () => randomNormal(this.standardDeviation))
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
   * In training and validation modes logits are calculated for every token,
   * since the loss is measured on the prediction of the next token at every
   * position in the sequence. In generation mode only the last token's
   * logits are needed to predict the next token, so the other tokens are
   * skipped. The logits are returned unchanged in every mode; turning them
   * into probabilities is left to the caller (e.g. Generator).
   *
   * @param embeddings - Token embeddings: one row per token, each row embeddingSize values wide.
   * @returns The logits, each row vocabSize values wide: in training and validation modes one row per token, in token order; in generation mode a single row for the last token.
   */
  calculate(embeddings: number[][]): number[][] {
    const selectedEmbeddings = this.mode === Hyperparameters.GENERATION_MODE ? embeddings.slice(-1) : embeddings;

    const logits = selectedEmbeddings.map((embedding) =>
      this.weights.map((weightRow) =>
        weightRow.reduce((sum, weight, index) => sum + weight * embedding[index], 0)
      )
    );

    // Print the whole output in verbose mode; otherwise just its dimensions.
    if (DEBUG.LINEAR_OUTPUT_LAYER) {
      console.log("LinearOutputLayer| Calculation complete.");
      if (DEBUG.VERBOSE) {
        console.log("LinearOutputLayer| Logits (one row per token):");
        console.table(logits);
      } else {
        console.log(`LinearOutputLayer| Logits. Rows: ${logits.length}; columns: ${logits[0]?.length ?? 0}`);
      }
    }

    return logits;
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
