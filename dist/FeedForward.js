import Hyperparameters from "./Hyperparameters.js";
import DEBUG from "./Debug.js";
/**
 * The position-wise feed forward network used in each GPT transformer
 * block: a linear layer that expands each token vector from embeddingSize
 * to 4 * embeddingSize values, a GELU activation, then a second linear
 * layer that projects it back down to embeddingSize values.
 *
 * Input and output are both batchSize x contextLength x embeddingSize.
 * The linear layers act only on the last dimension, so every token in
 * every sequence of the batch goes through the same weights independently.
 * The shapes as data flows through the network are:
 *
 * - First linear layer: batchSize x contextLength x embeddingSize in,
 *   batchSize x contextLength x (4 * embeddingSize) out.
 * - GELU: batchSize x contextLength x (4 * embeddingSize) in and out.
 * - Second linear layer: batchSize x contextLength x (4 * embeddingSize)
 *   in, batchSize x contextLength x embeddingSize out.
 */
class FeedForward {
    /** Number of sequences in each batch. */
    batchSize;
    /** Maximum number of tokens in each sequence. */
    contextLength;
    /** Width of each token vector at the input and output of the network. */
    embeddingSize;
    /** Width of each token vector between the two linear layers: 4 * embeddingSize. */
    hiddenSize;
    /** First linear layer weights built by build(): embeddingSize rows by hiddenSize columns. */
    firstLayerWeights = [];
    /** First linear layer biases built by build(): one per hidden unit, hiddenSize values. */
    firstLayerBiases = [];
    /** Second linear layer weights built by build(): hiddenSize rows by embeddingSize columns. */
    secondLayerWeights = [];
    /** Second linear layer biases built by build(): one per output unit, embeddingSize values. */
    secondLayerBiases = [];
    /**
     * Creates a FeedForward, copying batchSize, contextLength and
     * embeddingSize from the given Hyperparameters so this instance always
     * matches the values used elsewhere in the application, then builds the
     * network.
     *
     * @param hyperparameters - Source of batchSize, contextLength and embeddingSize. Defaults to a new Hyperparameters instance.
     */
    constructor(hyperparameters = new Hyperparameters()) {
        this.batchSize = hyperparameters.batchSize;
        this.contextLength = hyperparameters.contextLength;
        this.embeddingSize = hyperparameters.embeddingSize;
        this.hiddenSize = 4 * this.embeddingSize;
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
        this.firstLayerBiases = buildBiases(this.embeddingSize, this.hiddenSize);
        this.secondLayerWeights = buildMatrix(this.hiddenSize, this.embeddingSize);
        this.secondLayerBiases = buildBiases(this.hiddenSize, this.embeddingSize);
        if (DEBUG.FEED_FORWARD) {
            console.log(`FeedForward| Network built. Input: ${this.batchSize} x ${this.contextLength} x ${this.embeddingSize}; ` +
                `hidden: ${this.batchSize} x ${this.contextLength} x ${this.hiddenSize}; ` +
                `output: ${this.batchSize} x ${this.contextLength} x ${this.embeddingSize}`);
            console.log("FeedForward| First layer weights:");
            console.table(this.firstLayerWeights);
            console.log("FeedForward| Second layer weights:");
            console.table(this.secondLayerWeights);
        }
    }
    /**
     * Runs the input through the network: first linear layer, GELU, then
     * second linear layer. Each token vector is processed independently.
     *
     * @param input - A batchSize x contextLength x embeddingSize tensor: one matrix per sequence, one row per token. Sequences may be shorter than contextLength.
     * @returns A tensor with the same batchSize x contextLength x embeddingSize shape as the input.
     */
    calculate(input) {
        /*
         * A linear layer applied to a single token vector: each output value is
         * the dot product of the vector with one column of the weight matrix,
         * plus that column's bias.
         */
        const linear = (vector, weights, biases) => biases.map((bias, column) => vector.reduce((sum, value, row) => sum + value * weights[row][column], bias));
        /*
         * GELU using the tanh approximation from the original GPT-2 code:
         * 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3))).
         * Unlike ReLU it is smooth and lets small negative values through,
         * which gives non-zero gradients for negative inputs.
         */
        const gelu = (x) => 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x ** 3)));
        const output = input.map((sequence) => sequence.map((token) => {
            const hidden = linear(token, this.firstLayerWeights, this.firstLayerBiases).map(gelu);
            return linear(hidden, this.secondLayerWeights, this.secondLayerBiases);
        }));
        if (DEBUG.FEED_FORWARD) {
            output.forEach((sequence, index) => {
                console.log(`FeedForward| Output for sequence ${index} (one row per token):`);
                console.table(sequence);
            });
        }
        return output;
    }
}
export default FeedForward;
