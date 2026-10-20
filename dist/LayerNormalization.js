import Hyperparameters from "./Hyperparameters.js";
import { runningMean } from "./Matrix.js";
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
    /** Gradients of the loss with respect to scale, set by calculateGradients(). Empty until then. */
    scaleGradients = new Float32Array(0);
    /** Gradients of the loss with respect to shift, set by calculateGradients(). Empty until then. */
    shiftGradients = new Float32Array(0);
    /** The number of windows averaged into scaleGradients and shiftGradients. */
    gradientCount = 0;
    /** The normalized values (x̂, before scale and shift) of each token in the most recent update(), stored for backpropagation. */
    normalized = [];
    /** The standard deviation (σ, including epsilon) of each token in the most recent update(), stored for backpropagation. */
    standardDeviations = [];
    /** The errors given to the most recent backpropagate(), stored for calculating the gradients. */
    outputErrors = [];
    /**
     * Creates a LayerNormalization, copying embeddingSize from the given
     * Hyperparameters. As in GPT-2, scale starts as all ones and shift as all
     * zeros, so before training the layer just normalizes its input.
     *
     * @param hyperparameters - Source of embeddingSize. Defaults to a new Hyperparameters instance.
     */
    constructor(hyperparameters = new Hyperparameters()) {
        this.embeddingSize = hyperparameters.embeddingSize;
        this.scale = new Float32Array(this.embeddingSize).fill(1);
        // A new Float32Array is filled with zeros, so the shift starts at 0.
        this.shift = new Float32Array(this.embeddingSize);
    }
    /**
     * Normalizes every token vector in a sequence independently, so each has
     * a mean of 0 and a variance of 1, by subtracting the vector's mean from
     * each of its values and dividing by the square root of its variance plus
     * epsilon. Each normalized value is then multiplied by its dimension's
     * scale and has its shift added. The same scale and shift are used for
     * every token.
     *
     * The whole sequence is passed in at once, rather than one token at a
     * time, so that during training this instance can keep the values it
     * needs for backpropagation for every token, not just the last one.
     *
     * @param vectors - The token vectors to normalize: one row per token, each row embeddingSize values wide.
     * @returns A new matrix of the same shape holding the normalized, scaled and shifted vectors, in token order.
     * @throws RangeError if any row is not embeddingSize long, since there would be no scale and shift for some values.
     */
    update(vectors) {
        const output = [];
        // Each token's normalized values and standard deviation are stored for backpropagate().
        this.normalized = [];
        this.standardDeviations = [];
        for (let token = 0; token < vectors.length; token++) {
            const values = vectors[token];
            if (values.length !== this.embeddingSize) {
                throw new RangeError(`LayerNormalization| Token ${token}: expected ${this.embeddingSize} values; got ${values.length}`);
            }
            const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
            /*
             * The population variance (dividing by n rather than n - 1), which is
             * what PyTorch's nn.LayerNorm and GPT-2 use.
             */
            const variance = values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length;
            const standardDeviation = Math.sqrt(variance + LayerNormalization.epsilon);
            // Normalize each value, then apply that dimension's own scale and shift.
            const normalizedValues = [];
            const normalizedVector = [];
            for (let i = 0; i < values.length; i++) {
                const normalized = (values[i] - mean) / standardDeviation;
                normalizedValues.push(normalized);
                normalizedVector.push(this.scale[i] * normalized + this.shift[i]);
            }
            output.push(normalizedVector);
            this.normalized.push(normalizedValues);
            this.standardDeviations.push(standardDeviation);
        }
        return output;
    }
    /**
     * Passes the errors at the output back to the input, and stores them for
     * calculateGradients(). Each token is independent, so each is handled
     * separately.
     *
     * The output is y = scale * x̂ + shift, so the errors at the normalized
     * values are x̂Error = outputError * scale. Getting from there to the
     * input x is harder, because each x̂[i] = (x[i] - mean) / σ depends on
     * every input value through the mean and σ. Working through the
     * derivatives gives, with n = embeddingSize:
     *
     *     inputError = (1 / σ) * (x̂Error - mean(x̂Error) - x̂ * mean(x̂Error * x̂))
     *
     * The three terms are the three paths from x[i] to the loss:
     * - x̂Error / σ: the direct path, x[i] changing x̂[i];
     * - - mean(x̂Error) / σ: through the mean. Raising any x[i] raises the mean,
     *   which lowers every x̂[j] equally;
     * - - x̂ * mean(x̂Error * x̂) / σ: through σ. Raising x[i] changes the
     *   spread, which rescales every x̂[j] in proportion to its size.
     * A consequence is that the input errors of each token sum to 0: adding
     * the same amount to every input value changes nothing after normalization.
     *
     * @param outputErrors - The errors at the output: one row per token, each embeddingSize values wide.
     * @returns The errors at the input: one row per token, each embeddingSize values wide.
     * @throws RangeError if outputErrors does not have one row per token given to the most recent update().
     */
    backpropagate(outputErrors) {
        if (outputErrors.length !== this.normalized.length) {
            throw new RangeError(`LayerNormalization| Expected errors for ${this.normalized.length} tokens (from the most recent update()); got ${outputErrors.length}`);
        }
        this.outputErrors = outputErrors;
        const inputErrors = [];
        for (let token = 0; token < outputErrors.length; token++) {
            const normalized = this.normalized[token];
            const n = normalized.length;
            // Errors at the normalized values: each output is scale[i] times x̂[i] plus shift[i].
            const normalizedErrors = outputErrors[token].map((error, i) => error * this.scale[i]);
            // The two means in the formula.
            const meanError = normalizedErrors.reduce((sum, error) => sum + error, 0) / n;
            const meanErrorTimesNormalized = normalizedErrors.reduce((sum, error, i) => sum + error * normalized[i], 0) / n;
            const standardDeviation = this.standardDeviations[token];
            inputErrors.push(normalizedErrors.map((error, i) => (error - meanError - normalized[i] * meanErrorTimesNormalized) / standardDeviation));
        }
        return inputErrors;
    }
    /**
     * Calculates the gradients of scale and shift from the normalized values
     * stored by update() and the errors stored by backpropagate(). Each output
     * is y[i] = scale[i] * x̂[i] + shift[i], so for one token scale[i]'s
     * gradient is outputError[i] * x̂[i], and shift[i]'s is outputError[i].
     * The same scale and shift are used for every token, so the gradients are
     * summed over the tokens.
     *
     * As in dnn, the first window of a batch (batchAverage false) replaces the
     * stored gradients, and each later window (batchAverage true) is folded
     * into their running mean.
     *
     * @param batchAverage - False for the first window of a batch; true to average this window into the stored gradients.
     */
    calculateGradients(batchAverage) {
        const scaleGradients = new Float32Array(this.embeddingSize);
        const shiftGradients = new Float32Array(this.embeddingSize);
        for (let token = 0; token < this.outputErrors.length; token++) {
            for (let i = 0; i < this.embeddingSize; i++) {
                scaleGradients[i] += this.outputErrors[token][i] * this.normalized[token][i];
                shiftGradients[i] += this.outputErrors[token][i];
            }
        }
        this.gradientCount = batchAverage && this.gradientCount > 0 ? this.gradientCount + 1 : 1;
        this.scaleGradients = runningMean(this.scaleGradients, scaleGradients, this.gradientCount);
        this.shiftGradients = runningMean(this.shiftGradients, shiftGradients, this.gradientCount);
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
