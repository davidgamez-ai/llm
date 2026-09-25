/**
 * Layer normalization. Normalizes each token's activations to zero mean
 * and unit variance.
 */
class LayerNormalization {
    /**
     * Small constant added to the variance before taking its square root,
     * so normalization never divides by zero when a token's values are all
     * the same.
     */
    static epsilon = 0.00001;
    /**
     * Normalizes the values so they have a mean of 0 and a variance of 1, by
     * subtracting the mean from each value and dividing by the square root
     * of the variance plus epsilon.
     *
     * @param values - The values to normalize, e.g. one token's embedding vector.
     * @returns A new array, the same length as values, holding the normalized values. An empty input gives an empty result.
     */
    static calculate(values) {
        const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
        /*
         * The population variance (dividing by n rather than n - 1), which is
         * what PyTorch's nn.LayerNorm and GPT-2 use.
         */
        const variance = values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length;
        const standardDeviation = Math.sqrt(variance + LayerNormalization.epsilon);
        return values.map((value) => (value - mean) / standardDeviation);
    }
}
export default LayerNormalization;
