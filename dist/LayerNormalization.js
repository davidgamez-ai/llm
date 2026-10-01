import Hyperparameters from "./Hyperparameters.js";
/**
 * Layer normalization, as in GPT-2 and PyTorch's nn.LayerNorm. Normalizes
 * each token's activations to zero mean and unit variance, then applies a
 * trainable scale and shift so the model can learn the best mean and
 * variance for each dimension. Each place in the model that normalizes
 * holds its own instance, so each has its own scale and shift.
 */
class LayerNormalization {
    /**
     * Small constant added to the variance before taking its square root,
     * so normalization never divides by zero when a token's values are all
     * the same.
     */
    static epsilon = 0.00001;
    /** Width of each vector being normalized, i.e. the number of values in scale and shift. */
    embeddingSize;
    /** Trainable scale (gamma), multiplied into each normalized value: embeddingSize values, initialized to 1. */
    scale;
    /** Trainable shift (beta), added to each scaled value: embeddingSize values, initialized to 0. */
    shift;
    /**
     * Creates a LayerNormalization, copying embeddingSize from the given
     * Hyperparameters. As in GPT-2, scale starts as all ones and shift as all
     * zeros, so before training the layer just normalizes its input.
     *
     * @param hyperparameters - Source of embeddingSize. Defaults to a new Hyperparameters instance.
     */
    constructor(hyperparameters = new Hyperparameters()) {
        this.embeddingSize = hyperparameters.embeddingSize;
        this.scale = new Array(this.embeddingSize).fill(1);
        this.shift = new Array(this.embeddingSize).fill(0);
    }
    /**
     * Normalizes the values so they have a mean of 0 and a variance of 1, by
     * subtracting the mean from each value and dividing by the square root
     * of the variance plus epsilon. Each normalized value is then multiplied
     * by its scale and has its shift added.
     *
     * @param values - The values to normalize, e.g. one token's embedding vector. Must be embeddingSize values long.
     * @returns A new array, the same length as values, holding the normalized, scaled and shifted values.
     * @throws RangeError if values is not embeddingSize long, since there would be no scale and shift for some values.
     */
    calculate(values) {
        if (values.length !== this.embeddingSize) {
            throw new RangeError(`LayerNormalization| Expected ${this.embeddingSize} values; got ${values.length}`);
        }
        const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
        /*
         * The population variance (dividing by n rather than n - 1), which is
         * what PyTorch's nn.LayerNorm and GPT-2 use.
         */
        const variance = values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length;
        const standardDeviation = Math.sqrt(variance + LayerNormalization.epsilon);
        return values.map((value, index) => this.scale[index] * ((value - mean) / standardDeviation) + this.shift[index]);
    }
    /**
     * Counts the trainable parameters: every value in scale and shift.
     *
     * @returns The total number of scale and shift values, i.e. 2 * embeddingSize.
     */
    getParameterCount() {
        return this.scale.length + this.shift.length;
    }
}
export default LayerNormalization;
