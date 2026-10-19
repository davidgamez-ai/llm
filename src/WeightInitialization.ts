/*
 * Helpers for initializing weights as GPT-2 does (Hugging Face
 * GPT2PreTrainedModel._init_weights and the GPT-2 paper):
 *
 * - The weights of every linear layer and embedding are drawn from a normal
 *   distribution with mean 0 and standard deviation 0.02.
 * - The two residual projections in each transformer block (the attention
 *   output projection and the second feed forward layer) are drawn with that
 *   standard deviation divided by sqrt(2 * numberTransformerBlocks).
 * - Every bias starts at 0.
 * - Layer normalization scale starts at 1 and shift at 0.
 */

import Matrix from "./Matrix.js";

/**
 * Draws one sample from a normal distribution with mean 0 and the given
 * standard deviation, using the Box-Muller transform to turn two uniform
 * samples into a standard normal sample. 1 - random() lies in (0, 1], so
 * Math.log never receives 0.
 *
 * @param random - Source of uniform random values in [0, 1), e.g. Hyperparameters.random.
 * @param standardDeviation - The standard deviation of the distribution.
 * @returns One sample.
 */
export function randomNormal(random: () => number, standardDeviation: number): number {
  const u1 = 1 - random();
  const u2 = random();
  return standardDeviation * Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

/**
 * Builds a rows x columns matrix of samples from a normal distribution with
 * mean 0 and the given standard deviation, filled row by row.
 *
 * @param rows - The number of rows.
 * @param columns - The number of columns.
 * @param standardDeviation - The standard deviation of the distribution.
 * @param random - Source of uniform random values in [0, 1), e.g. Hyperparameters.random.
 * @returns The new matrix.
 */
export function randomNormalMatrix(
  rows: number,
  columns: number,
  standardDeviation: number,
  random: () => number
): Matrix {
  const matrix = new Matrix(rows, columns);
  for (let i = 0; i < matrix.size; i++) {
    matrix.data[i] = randomNormal(random, standardDeviation);
  }
  return matrix;
}

/**
 * The standard deviation GPT-2 uses for a residual projection: a linear
 * layer whose output is added straight to the residual stream. Every
 * transformer block adds two such outputs to the stream (attention and feed
 * forward), so the GPT-2 paper scales their initial weights by
 * 1 / sqrt(2 * numberTransformerBlocks), which stops the variance of the
 * residual stream growing with the model's depth.
 *
 * @param standardDeviation - The standard deviation used for other linear layers (0.02 in GPT-2).
 * @param numberTransformerBlocks - The number of transformer blocks in the model.
 * @returns standardDeviation / sqrt(2 * numberTransformerBlocks).
 */
export function residualProjectionStandardDeviation(standardDeviation: number, numberTransformerBlocks: number): number {
  return standardDeviation / Math.sqrt(2 * numberTransformerBlocks);
}
