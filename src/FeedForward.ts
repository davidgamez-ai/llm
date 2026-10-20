import Hyperparameters from "./Hyperparameters.js";
import { randomNormalMatrix, residualProjectionStandardDeviation } from "./WeightInitialization.js";
import Matrix, { vectorTimesMatrix, matrixTimesVector, addOuterProduct, addToVector, runningMean } from "./Matrix.js";
import DEBUG from "./Debug.js";

/** sqrt(2 / pi), used by the tanh approximation of GELU. */
const GELU_SCALE: number = Math.sqrt(2 / Math.PI);

/**
 * GELU, using the tanh approximation from the original GPT-2 code:
 *
 *     gelu(x) = 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))
 *
 * Unlike ReLU it is smooth and lets small negative values through, which
 * gives non-zero gradients for negative inputs.
 *
 * @param x - The input value.
 * @returns GELU of x.
 */
export function gelu(x: number): number {
  return 0.5 * x * (1 + Math.tanh(GELU_SCALE * (x + 0.044715 * x ** 3)));
}

/**
 * The slope of gelu at x, used to pass errors back through it. Writing
 * u = sqrt(2 / pi) * (x + 0.044715 * x^3), so that gelu(x) = 0.5 * x * (1 + tanh(u)),
 * the product rule and the chain rule (with d tanh(u)/du = 1 - tanh(u)^2) give
 *
 *     gelu'(x) = 0.5 * (1 + tanh(u)) + 0.5 * x * (1 - tanh(u)^2) * sqrt(2 / pi) * (1 + 3 * 0.044715 * x^2)
 *
 * The first term is the slope of x with tanh(u) held fixed; the second is
 * the effect of x on tanh(u).
 *
 * @param x - The input value.
 * @returns The derivative of gelu at x.
 */
export function geluDerivative(x: number): number {
  const tanhU = Math.tanh(GELU_SCALE * (x + 0.044715 * x ** 3));
  return 0.5 * (1 + tanhU) + 0.5 * x * (1 - tanhU * tanhU) * GELU_SCALE * (1 + 3 * 0.044715 * x * x);
}

/**
 * The position-wise feed forward network used in each GPT transformer
 * block: a linear layer that expands a token vector from embeddingSize
 * to 4 * embeddingSize values, a GELU activation, then a second linear
 * layer that projects it back down to embeddingSize values.
 *
 * The network is "position-wise": every token in a sequence goes through
 * the same weights independently, and tokens never interact. update()
 * takes the whole sequence at once so that, during training, the instance
 * can keep the values it needs for backpropagation for every token.
 * The shapes of each token vector as data flows through the network are:
 *
 * - First linear layer: embeddingSize in, 4 * embeddingSize out.
 * - GELU: 4 * embeddingSize in and out.
 * - Second linear layer: 4 * embeddingSize in, embeddingSize out.
 */
class FeedForward {
  /** Width of the token vector at the input and output of the network. */
  embeddingSize: number;

  /** Width of the token vector between the two linear layers: 4 * embeddingSize. */
  hiddenSize: number;

  /** True if the linear layers add a bias to their outputs; false if they do not. */
  bias: boolean;

  /** First linear layer weights built by build(): embeddingSize rows by hiddenSize columns. */
  firstLayerWeights: Matrix = new Matrix(0, 0);

  /** First linear layer biases built by build(): one per hidden unit, hiddenSize values. Empty when bias is false. */
  firstLayerBiases: Float32Array = new Float32Array(0);

  /** Second linear layer weights built by build(): hiddenSize rows by embeddingSize columns. */
  secondLayerWeights: Matrix = new Matrix(0, 0);

  /** Second linear layer biases built by build(): one per output unit, embeddingSize values. Empty when bias is false. */
  secondLayerBiases: Float32Array = new Float32Array(0);

  /** Standard deviation for linear layer weights (0.02 in GPT-2); the second layer uses it scaled by 1 / sqrt(2 * numberTransformerBlocks). */
  linearLayerStandardDeviation: number;

  /** Number of transformer blocks in the model, used to scale the second layer's initial weights. */
  numberTransformerBlocks: number;

  /** Gradients of firstLayerWeights, set by calculateGradients(). Empty until then. */
  firstLayerWeightGradients: Matrix = new Matrix(0, 0);

  /** Gradients of firstLayerBiases, set by calculateGradients(). Empty until then, and always empty when bias is false. */
  firstLayerBiasGradients: Float32Array = new Float32Array(0);

  /** Gradients of secondLayerWeights, set by calculateGradients(). Empty until then. */
  secondLayerWeightGradients: Matrix = new Matrix(0, 0);

  /** Gradients of secondLayerBiases, set by calculateGradients(). Empty until then, and always empty when bias is false. */
  secondLayerBiasGradients: Float32Array = new Float32Array(0);

  /** The number of windows averaged into the gradients. */
  gradientCount: number = 0;

  /** The token vectors given to the most recent update(), stored for calculating the gradients. */
  private inputs: number[][] = [];

  /** Each token's first layer output before GELU, stored for backpropagating through GELU. */
  private preActivations: number[][] = [];

  /** Each token's first layer output after GELU, stored for calculating the second layer's gradients. */
  private activations: number[][] = [];

  /** The errors given to the most recent backpropagate(), stored for calculating the second layer's gradients. */
  private outputErrors: number[][] = [];

  /** Each token's errors at the first layer's output (before GELU), stored for calculating the first layer's gradients. */
  private preActivationErrors: number[][] = [];

  /** Draws from the Hyperparameters' shared random number stream, which is seeded when Hyperparameters.seed is set. */
  private readonly random: () => number;

  /**
   * Creates a FeedForward, copying embeddingSize, feedForwardBias,
   * linearLayerStandardDeviation and numberTransformerBlocks from the given
   * Hyperparameters so this instance always matches the values used
   * elsewhere in the application, then builds the network.
   *
   * @param hyperparameters - Source of embeddingSize, feedForwardBias, linearLayerStandardDeviation, numberTransformerBlocks and the random number stream. Defaults to a new Hyperparameters instance.
   */
  constructor(hyperparameters: Hyperparameters = new Hyperparameters()) {
    this.embeddingSize = hyperparameters.embeddingSize;
    this.hiddenSize = 4 * this.embeddingSize;
    this.bias = hyperparameters.feedForwardBias;
    this.linearLayerStandardDeviation = hyperparameters.linearLayerStandardDeviation;
    this.numberTransformerBlocks = hyperparameters.numberTransformerBlocks;
    this.random = () => hyperparameters.random();

    this.build();
  }

  /**
   * Builds the weights and biases of the two linear layers and stores them
   * on the instance. The first layer maps embeddingSize inputs to hiddenSize
   * (4 * embeddingSize) outputs; the second maps hiddenSize inputs back to
   * embeddingSize outputs. The GELU activation between them has no
   * parameters, so nothing is built for it. Initialized as in GPT-2 (Hugging
   * Face GPT2PreTrainedModel._init_weights, for GPT-2's mlp.c_fc and
   * mlp.c_proj):
   * - First layer weights are drawn from a normal distribution with mean 0
   *   and standard deviation linearLayerStandardDeviation (0.02 by default).
   * - Second layer weights use that standard deviation divided by
   *   sqrt(2 * numberTransformerBlocks), because the second layer is a
   *   residual projection: its output is added straight to the residual
   *   stream (see WeightInitialization).
   * - Biases are set to 0. They are only built when bias is true; otherwise
   *   the bias arrays are left empty.
   *
   * @returns void. The resulting weights and biases are stored on the instance.
   */
  build(): void {
    const secondLayerStandardDeviation = residualProjectionStandardDeviation(
      this.linearLayerStandardDeviation,
      this.numberTransformerBlocks
    );

    this.firstLayerWeights = randomNormalMatrix(this.embeddingSize, this.hiddenSize, this.linearLayerStandardDeviation, this.random);
    this.secondLayerWeights = randomNormalMatrix(this.hiddenSize, this.embeddingSize, secondLayerStandardDeviation, this.random);
    // A new Float32Array is filled with zeros, so the biases start at 0.
    this.firstLayerBiases = new Float32Array(this.bias ? this.hiddenSize : 0);
    this.secondLayerBiases = new Float32Array(this.bias ? this.embeddingSize : 0);

    if (DEBUG.FEED_FORWARD) {
      console.log(
        `FeedForward| Network built. Input: ${this.embeddingSize}; ` +
        `hidden: ${this.hiddenSize}; output: ${this.embeddingSize}`
      );
      console.log("FeedForward| First layer weights:");
      console.table(this.firstLayerWeights.toArrays());
      console.log("FeedForward| Second layer weights:");
      console.table(this.secondLayerWeights.toArrays());
    }
  }

  /**
   * Runs each token vector in a sequence through the network independently:
   * first linear layer, GELU, then second linear layer. Every token uses the
   * same weights.
   *
   * @param tokens - The token vectors: one row per token, each row embeddingSize values wide.
   * @returns The transformed token vectors: one row per token, in token order, each embeddingSize values wide.
   */
  update(tokens: number[][]): number[][] {
    /*
     * Each linear layer multiplies the vector by its weight matrix and adds
     * its biases (which are empty when bias is false): each output value is
     * the dot product of the vector with one column of the weight matrix,
     * plus that column's bias.
     *
     * The input, the first layer's output before GELU (the pre-activation)
     * and after it (the activation) are stored for every token, because
     * backpropagation and the gradients need them.
     */
    this.inputs = tokens;
    this.preActivations = [];
    this.activations = [];

    const output: number[][] = [];
    for (const token of tokens) {
      const preActivation = vectorTimesMatrix(token, this.firstLayerWeights, this.firstLayerBiases);
      const activation = preActivation.map(gelu);
      this.preActivations.push(preActivation);
      this.activations.push(activation);
      output.push(vectorTimesMatrix(activation, this.secondLayerWeights, this.secondLayerBiases));
    }

    if (DEBUG.FEED_FORWARD) {
      console.log("FeedForward| Output (one row per token):");
      console.table(output);
    }

    return output;
  }

  /**
   * Passes the errors at the output back through the network to its input,
   * one token at a time, and stores what calculateGradients() needs. This is
   * the forward pass in reverse:
   *
   * 1. Second linear layer: each activation contributed to every output
   *    through one row of secondLayerWeights, so its error is the dot product
   *    of that row with the output errors: the weights transposed, which is
   *    matrixTimesVector because the weights are stored input x output.
   * 2. GELU: each pre-activation p only affects its own activation, so its
   *    error is the activation's error times GELU's slope at p.
   * 3. First linear layer: as in step 1, through firstLayerWeights.
   *
   * @param outputErrors - The errors at the output: one row per token, each embeddingSize values wide.
   * @returns The errors at the input: one row per token, each embeddingSize values wide.
   * @throws RangeError if outputErrors does not have one row per token given to the most recent update().
   */
  backpropagate(outputErrors: number[][]): number[][] {
    if (outputErrors.length !== this.inputs.length) {
      throw new RangeError(
        `FeedForward| Expected errors for ${this.inputs.length} tokens (from the most recent update()); got ${outputErrors.length}`
      );
    }
    this.outputErrors = outputErrors;
    this.preActivationErrors = [];

    const inputErrors: number[][] = [];
    for (let token = 0; token < outputErrors.length; token++) {
      const activationErrors = matrixTimesVector(this.secondLayerWeights, outputErrors[token]);
      const preActivationErrors = activationErrors.map((error, i) => error * geluDerivative(this.preActivations[token][i]));
      this.preActivationErrors.push(preActivationErrors);
      inputErrors.push(matrixTimesVector(this.firstLayerWeights, preActivationErrors));
    }
    return inputErrors;
  }

  /**
   * Calculates the gradients of both layers' weights and biases from the
   * values stored by update() and the errors stored by backpropagate(). For
   * each linear layer, weight (i, j) multiplied input i to produce output j,
   * so for one token its gradient is input i times output j's error; each
   * bias is added straight to its output, so its gradient is that output's
   * error. The same weights are used for every token, so the gradients are
   * summed over the tokens:
   *
   *     secondLayerWeightGradients = sum over tokens of activation ⊗ outputError
   *     secondLayerBiasGradients   = sum over tokens of outputError
   *     firstLayerWeightGradients  = sum over tokens of input ⊗ preActivationError
   *     firstLayerBiasGradients    = sum over tokens of preActivationError
   *
   * The bias gradients are left empty when bias is false. As in dnn, the
   * first window of a batch (batchAverage false) replaces the stored
   * gradients, and each later window (batchAverage true) is folded into
   * their running mean.
   *
   * @param batchAverage - False for the first window of a batch; true to average this window into the stored gradients.
   */
  calculateGradients(batchAverage: boolean): void {
    const firstLayerWeightGradients = new Matrix(this.embeddingSize, this.hiddenSize);
    const firstLayerBiasGradients = new Float32Array(this.firstLayerBiases.length);
    const secondLayerWeightGradients = new Matrix(this.hiddenSize, this.embeddingSize);
    const secondLayerBiasGradients = new Float32Array(this.secondLayerBiases.length);

    for (let token = 0; token < this.outputErrors.length; token++) {
      addOuterProduct(secondLayerWeightGradients, this.activations[token], this.outputErrors[token]);
      addOuterProduct(firstLayerWeightGradients, this.inputs[token], this.preActivationErrors[token]);
      if (this.bias) {
        addToVector(secondLayerBiasGradients, this.outputErrors[token]);
        addToVector(firstLayerBiasGradients, this.preActivationErrors[token]);
      }
    }

    this.gradientCount = batchAverage && this.gradientCount > 0 ? this.gradientCount + 1 : 1;
    this.firstLayerWeightGradients = runningMean(this.firstLayerWeightGradients, firstLayerWeightGradients, this.gradientCount);
    this.firstLayerBiasGradients = runningMean(this.firstLayerBiasGradients, firstLayerBiasGradients, this.gradientCount);
    this.secondLayerWeightGradients = runningMean(this.secondLayerWeightGradients, secondLayerWeightGradients, this.gradientCount);
    this.secondLayerBiasGradients = runningMean(this.secondLayerBiasGradients, secondLayerBiasGradients, this.gradientCount);
  }

  /**
   * Counts the trainable parameters in the network: the weights of both
   * linear layers plus their biases. The biases are empty, and so add
   * nothing, when bias is false.
   *
   * @returns The total number of weights and biases currently stored on the instance.
   */
  getParameterCount(): number {
    return (
      this.firstLayerWeights.size +
      this.firstLayerBiases.length +
      this.secondLayerWeights.size +
      this.secondLayerBiases.length
    );
  }
}

export default FeedForward;
