import Hyperparameters from "./Hyperparameters.js";
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
    firstLayerWeights = [];
    /** First linear layer biases built by build(): one per hidden unit, hiddenSize values. Empty when bias is false. */
    firstLayerBiases = [];
    /** Second linear layer weights built by build(): hiddenSize rows by embeddingSize columns. */
    secondLayerWeights = [];
    /** Second linear layer biases built by build(): one per output unit, embeddingSize values. Empty when bias is false. */
    secondLayerBiases = [];
    /**
     * Creates a FeedForward, copying embeddingSize and bias from the given
     * Hyperparameters so this instance always matches the values used
     * elsewhere in the application, then builds the network.
     *
     * @param hyperparameters - Source of embeddingSize and bias. Defaults to a new Hyperparameters instance.
     */
    constructor(hyperparameters = new Hyperparameters()) {
        this.embeddingSize = hyperparameters.embeddingSize;
        this.hiddenSize = 4 * this.embeddingSize;
        this.bias = hyperparameters.bias;
        this.build();
    }
    /**
     * Builds the weights and biases of the two linear layers and stores them
     * on the instance. The first layer maps embeddingSize inputs to hiddenSize
     * (4 * embeddingSize) outputs; the second maps hiddenSize inputs back to
     * embeddingSize outputs. The GELU activation between them has no
     * parameters, so nothing is built for it. Weights and biases are
     * initialized the same way as PyTorch's nn.Linear: drawn uniformly from
     * [-1/sqrt(fanIn), 1/sqrt(fanIn)), where fanIn is the layer's input width.
     * Biases are only built when bias is true; otherwise the bias arrays are
     * left empty.
     *
     * @returns void. The resulting weights and biases are stored on the instance.
     */
    build() {
        /*
         * A fresh rows x columns matrix of random values in [-bound, bound),
         * where bound = 1 / sqrt(rows). rows is the layer's fan in, so scaling
         * by it keeps the variance of each layer's output roughly independent
         * of its input width.
         */
        const buildMatrix = (rows, columns) => {
            const bound = 1 / Math.sqrt(rows);
            return Array.from({ length: rows }, () => Array.from({ length: columns }, () => (Math.random() * 2 - 1) * bound));
        };
        // A bias vector with the same bound as the matching weight matrix, as in nn.Linear.
        const buildBiases = (fanIn, length) => {
            const bound = 1 / Math.sqrt(fanIn);
            return Array.from({ length }, () => (Math.random() * 2 - 1) * bound);
        };
        this.firstLayerWeights = buildMatrix(this.embeddingSize, this.hiddenSize);
        this.secondLayerWeights = buildMatrix(this.hiddenSize, this.embeddingSize);
        this.firstLayerBiases = this.bias ? buildBiases(this.embeddingSize, this.hiddenSize) : [];
        this.secondLayerBiases = this.bias ? buildBiases(this.hiddenSize, this.embeddingSize) : [];
        if (DEBUG.FEED_FORWARD) {
            console.log(`FeedForward| Network built. Input: ${this.embeddingSize}; ` +
                `hidden: ${this.hiddenSize}; output: ${this.embeddingSize}`);
            console.log("FeedForward| First layer weights:");
            console.table(this.firstLayerWeights);
            console.log("FeedForward| Second layer weights:");
            console.table(this.secondLayerWeights);
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
         * A linear layer applied to a single vector: each output value is the
         * dot product of the vector with one column of the weight matrix, plus
         * that column's bias when bias is true. The dot product starts from the
         * bias, or from 0 when there is no bias.
         */
        const linear = (vector, weights, biases) => weights[0].map((_, column) => vector.reduce((sum, value, row) => sum + value * weights[row][column], this.bias ? biases[column] : 0));
        /*
         * GELU using the tanh approximation from the original GPT-2 code:
         * 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3))).
         * Unlike ReLU it is smooth and lets small negative values through,
         * which gives non-zero gradients for negative inputs.
         */
        const gelu = (x) => 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x ** 3)));
        const hidden = linear(token, this.firstLayerWeights, this.firstLayerBiases).map(gelu);
        const output = linear(hidden, this.secondLayerWeights, this.secondLayerBiases);
        if (DEBUG.FEED_FORWARD) {
            console.log("FeedForward| Output:");
            console.table(output);
        }
        return output;
    }
}
export default FeedForward;
