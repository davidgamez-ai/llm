import Hyperparameters from "./Hyperparameters.js";
import { randomNormalMatrix } from "./WeightInitialization.js";
import Matrix, { matrixTimesVector, vectorTimesMatrix, addOuterProduct, runningMean } from "./Matrix.js";
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

  /**
   * Gradients of the loss with respect to weights, the same shape, set by
   * calculateGradients(): this window's gradients, or their running mean
   * over the windows of the batch so far. Empty until calculateGradients()
   * is first called.
   */
  weightGradients: Matrix = new Matrix(0, 0);

  /** The number of windows averaged into weightGradients. */
  gradientCount: number = 0;

  /** The embeddings given to the most recent update(), stored for calculating the gradients. */
  private inputs: number[][] = [];

  /** The errors at the logits given to the most recent backpropagate(), stored for calculating the gradients. */
  private outputErrors: number[][] = [];

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
  update(embeddings: number[][]): number[][] {
    const selectedEmbeddings = this.mode === Hyperparameters.GENERATION_MODE ? embeddings.slice(-1) : embeddings;

    // Stored for calculateGradients(): each weight's gradient depends on the input it multiplied.
    this.inputs = selectedEmbeddings;

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
   * Passes the errors at the logits back to the layer's inputs, and stores
   * them for calculateGradients().
   *
   * Each logit is logit[v] = sum over e of weights(v, e) * input[e], so
   * input value e contributed to every logit through column e of the
   * weights. Its error is therefore the sum of every logit's error times
   * that weight:
   *
   *     inputError[e] = sum over v of outputError[v] * weights(v, e)
   *
   * This is the forward calculation with the weights transposed. The
   * weights are stored one row per vocabulary token (output x input), so
   * where the forward pass used matrixTimesVector, the backward pass uses
   * vectorTimesMatrix.
   *
   * @param outputErrors - The errors at the logits (from crossEntropyErrors): one row per token, each vocabSize values wide.
   * @returns The errors at the inputs: one row per token, each embeddingSize values wide.
   * @throws RangeError if outputErrors does not have one row per token given to the most recent update().
   */
  backpropagate(outputErrors: number[][]): number[][] {
    if (outputErrors.length !== this.inputs.length) {
      throw new RangeError(
        `LinearOutputLayer| Expected errors for ${this.inputs.length} tokens (from the most recent update()); got ${outputErrors.length}`
      );
    }
    this.outputErrors = outputErrors;

    return outputErrors.map((errors) => vectorTimesMatrix(errors, this.weights));
  }

  /**
   * Calculates the gradients of the weights from the inputs stored by
   * update() and the errors stored by backpropagate(). Weight (v, e)
   * multiplied input e to produce logit v, so for one token its gradient is
   * logit v's error times input e: the outer product of the errors with the
   * input. The same weights are used for every token, so the gradients are
   * summed over the tokens:
   *
   *     weightGradients(v, e) = sum over tokens of outputError[v] * input[e]
   *
   * As in dnn, the first window of a batch (batchAverage false) replaces the
   * stored gradients, and each later window (batchAverage true) is folded
   * into their running mean.
   *
   * @param batchAverage - False for the first window of a batch; true to average this window into the stored gradients.
   */
  calculateGradients(batchAverage: boolean): void {
    const weightGradients = new Matrix(this.vocabSize, this.embeddingSize);
    for (let token = 0; token < this.inputs.length; token++) {
      /*
       * The weights are stored output x input, so the errors index the rows
       * and the input indexes the columns: the arguments are the other way
       * round from the layers that store their weights input x output.
       */
      addOuterProduct(weightGradients, this.outputErrors[token], this.inputs[token]);
    }

    this.gradientCount = batchAverage && this.gradientCount > 0 ? this.gradientCount + 1 : 1;
    this.weightGradients = runningMean(this.weightGradients, weightGradients, this.gradientCount);
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
