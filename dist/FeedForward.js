import Hyperparameters from "./Hyperparameters.js";
import { randomNormalMatrix, residualProjectionStandardDeviation } from "./WeightInitialization.js";
import Matrix, { vectorTimesMatrix } from "./Matrix.js";
import DEBUG from "./Debug.js";
/**
 * The position-wise feed forward network used in each GPT transformer
 * block: a linear layer that expands a token vector from embeddingSize
 * to 4 * embeddingSize values, a GELU activation, then a second linear
 * layer that projects it back down to embeddingSize values.
 *
 * The network processes one token vector at a time. Every token in a
 * sequence goes through the same weights independently, so a whole
 * sequence is processed by calling calculate() on each of its tokens.
 * The shapes as data flows through the network are:
 *
 * - First linear layer: embeddingSize in, 4 * embeddingSize out.
 * - GELU: 4 * embeddingSize in and out.
 * - Second linear layer: 4 * embeddingSize in, embeddingSize out.
 */
class FeedForward {
    /** Width of the token vector at the input and output of the network. */
    embeddingSize;
    /** Width of the token vector between the two linear layers: 4 * embeddingSize. */
    hiddenSize;
    /** True if the linear layers add a bias to their outputs; false if they do not. */
    bias;
    /** First linear layer weights built by build(): embeddingSize rows by hiddenSize columns. */
    firstLayerWeights = new Matrix(0, 0);
    /** First linear layer biases built by build(): one per hidden unit, hiddenSize values. Empty when bias is false. */
    firstLayerBiases = new Float32Array(0);
    /** Second linear layer weights built by build(): hiddenSize rows by embeddingSize columns. */
    secondLayerWeights = new Matrix(0, 0);
    /** Second linear layer biases built by build(): one per output unit, embeddingSize values. Empty when bias is false. */
    secondLayerBiases = new Float32Array(0);
    /** Standard deviation for linear layer weights (0.02 in GPT-2); the second layer uses it scaled by 1 / sqrt(2 * numberTransformerBlocks). */
    linearLayerStandardDeviation;
    /** Number of transformer blocks in the model, used to scale the second layer's initial weights. */
    numberTransformerBlocks;
    /** Draws from the Hyperparameters' shared random number stream, which is seeded when Hyperparameters.seed is set. */
    random;
    /**
     * Creates a FeedForward, copying embeddingSize, feedForwardBias,
     * linearLayerStandardDeviation and numberTransformerBlocks from the given
     * Hyperparameters so this instance always matches the values used
     * elsewhere in the application, then builds the network.
     *
     * @param hyperparameters - Source of embeddingSize, feedForwardBias, linearLayerStandardDeviation, numberTransformerBlocks and the random number stream. Defaults to a new Hyperparameters instance.
     */
    constructor(hyperparameters = new Hyperparameters()) {
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
    build() {
        const secondLayerStandardDeviation = residualProjectionStandardDeviation(this.linearLayerStandardDeviation, this.numberTransformerBlocks);
        this.firstLayerWeights = randomNormalMatrix(this.embeddingSize, this.hiddenSize, this.linearLayerStandardDeviation, this.random);
        this.secondLayerWeights = randomNormalMatrix(this.hiddenSize, this.embeddingSize, secondLayerStandardDeviation, this.random);
        // A new Float32Array is filled with zeros, so the biases start at 0.
        this.firstLayerBiases = new Float32Array(this.bias ? this.hiddenSize : 0);
        this.secondLayerBiases = new Float32Array(this.bias ? this.embeddingSize : 0);
        if (DEBUG.FEED_FORWARD) {
            console.log(`FeedForward| Network built. Input: ${this.embeddingSize}; ` +
                `hidden: ${this.hiddenSize}; output: ${this.embeddingSize}`);
            console.log("FeedForward| First layer weights:");
            console.table(this.firstLayerWeights.toArrays());
            console.log("FeedForward| Second layer weights:");
            console.table(this.secondLayerWeights.toArrays());
        }
    }
    /**
     * Runs a single token vector through the network: first linear layer,
     * GELU, then second linear layer.
     *
     * @param token - The embedding of one token: embeddingSize values.
     * @returns The transformed token vector: embeddingSize values.
     */
    calculate(token) {
        /*
         * GELU using the tanh approximation from the original GPT-2 code:
         * 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3))).
         * Unlike ReLU it is smooth and lets small negative values through,
         * which gives non-zero gradients for negative inputs.
         */
        const gelu = (x) => 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x ** 3)));
        /*
         * Each linear layer multiplies the vector by its weight matrix and adds
         * its biases (which are empty when bias is false): each output value is
         * the dot product of the vector with one column of the weight matrix,
         * plus that column's bias.
         */
        const hidden = vectorTimesMatrix(token, this.firstLayerWeights, this.firstLayerBiases).map(gelu);
        const output = vectorTimesMatrix(hidden, this.secondLayerWeights, this.secondLayerBiases);
        if (DEBUG.FEED_FORWARD) {
            console.log("FeedForward| Output:");
            console.table(output);
        }
        return output;
    }
    /**
     * Counts the trainable parameters in the network: the weights of both
     * linear layers plus their biases. The biases are empty, and so add
     * nothing, when bias is false.
     *
     * @returns The total number of weights and biases currently stored on the instance.
     */
    getParameterCount() {
        return (this.firstLayerWeights.size +
            this.firstLayerBiases.length +
            this.secondLayerWeights.size +
            this.secondLayerBiases.length);
    }
}
export default FeedForward;
