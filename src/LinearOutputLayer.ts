import Hyperparameters from "./Hyperparameters.js";
import { randomNormalMatrix } from "./WeightInitialization.js";
import Matrix, { matrixTimesVector } from "./Matrix.js";
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
  weights: Matrix = new Matrix(0, 0);

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
   * instance. Initialized as in GPT-2 (Hugging Face
   * GPT2PreTrainedModel._init_weights), with values drawn from a normal
   * distribution with mean 0 and standard deviation standardDeviation (0.02
   * by default). GPT-2 itself ties this matrix to the token embedding, which
   * is initialized the same way; here the two are separate, as in the book.
   * Random values come from the Hyperparameters' shared stream, so the
   * matrix is reproducible when Hyperparameters.seed is set.
   *
   * @returns void. The resulting matrix is stored on the instance.
   */
  build(): void {
    /*
     * A small standard deviation keeps the initial logits close together:
     * each logit is a dot product of embeddingSize terms with a normalized
     * vector, so its standard deviation is about
     * standardDeviation * sqrt(embeddingSize), roughly 0.55 for 0.02 and 768.
     * The untrained model therefore gives every token a similar probability,
     * and the initial loss is close to ln(vocabSize), about 10.8.
     */
    this.weights = randomNormalMatrix(this.vocabSize, this.embeddingSize, this.standardDeviation, this.random);

    if (DEBUG.LINEAR_OUTPUT_LAYER) console.log(
      `LinearOutputLayer| Weight matrix built. Rows: ${this.weights.rows}; columns: ${this.weights.columns}`
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

    // weights has one row per vocabulary token, so each logit is one row of weights dotted with the embedding.
    const logits = selectedEmbeddings.map((embedding) => matrixTimesVector(this.weights, embedding));

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
    return this.weights.size;
  }
}

export default LinearOutputLayer;
