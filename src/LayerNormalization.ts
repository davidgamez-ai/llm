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
  static epsilon: number = 0.00001;

  /** Width of each vector being normalized, i.e. the number of values in scale and shift. */
  embeddingSize: number;

  /** Trainable scale (gamma), multiplied into each normalized value: embeddingSize values, initialized to 1. */
  scale: Float32Array;

  /** Trainable shift (beta), added to each scaled value: embeddingSize values, initialized to 0. */
  shift: Float32Array;

  /**
   * Creates a LayerNormalization, copying embeddingSize from the given
   * Hyperparameters. As in GPT-2, scale starts as all ones and shift as all
   * zeros, so before training the layer just normalizes its input.
   *
   * @param hyperparameters - Source of embeddingSize. Defaults to a new Hyperparameters instance.
   */
  constructor(hyperparameters: Hyperparameters = new Hyperparameters()) {
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
  update(vectors: number[][]): number[][] {
    const output: number[][] = [];

    for (let token = 0; token < vectors.length; token++) {
      const values = vectors[token];
      if (values.length !== this.embeddingSize) {
        throw new RangeError(
          `LayerNormalization| Token ${token}: expected ${this.embeddingSize} values; got ${values.length}`
        );
      }

      const mean = values.reduce((sum, value) => sum + value, 0) / values.length;

      /*
       * The population variance (dividing by n rather than n - 1), which is
       * what PyTorch's nn.LayerNorm and GPT-2 use.
       */
      const variance = values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length;

      const standardDeviation = Math.sqrt(variance + LayerNormalization.epsilon);

      // Normalize each value, then apply that dimension's own scale and shift.
      const normalizedVector: number[] = [];
      for (let i = 0; i < values.length; i++) {
        const normalized = (values[i] - mean) / standardDeviation;
        normalizedVector.push(this.scale[i] * normalized + this.shift[i]);
      }
      output.push(normalizedVector);
    }

    return output;
  }

  /**
   * Counts the trainable parameters: every value in scale and shift.
   *
   * @returns The total number of scale and shift values, i.e. 2 * embeddingSize.
   */
  getParameterCount(): number {
    return this.scale.length + this.shift.length;
  }
}

export default LayerNormalization;
