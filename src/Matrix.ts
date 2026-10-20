/**
 * A matrix of trainable weights, stored as a single Float32Array.
 *
 * The values are stored row by row ("row-major order"): row 0 first, then
 * row 1, and so on. The value at (row, column) is therefore at index
 * row * columns + column of data. For example, a 2 x 3 matrix
 *
 *     [ a b c ]
 *     [ d e f ]
 *
 * is stored as data = [a, b, c, d, e, f].
 *
 * Why not number[][]? A JavaScript number[][] is an array of separate
 * arrays, each holding 64-bit numbers scattered around memory. A
 * Float32Array is one contiguous block of 32-bit numbers, which:
 * - halves the memory needed (GPT-2 has 163 million parameters here, and
 *   training will need gradients and optimizer state for each one too);
 * - lets the CPU read values in order, which is much faster than jumping
 *   between separate arrays.
 *
 * 32-bit floats keep about 7 significant digits, which is what PyTorch uses
 * by default for training. JavaScript does the arithmetic itself in 64-bit,
 * so only the stored values are rounded.
 */
class Matrix {
  /** The number of rows. */
  readonly rows: number;

  /** The number of columns. */
  readonly columns: number;

  /** The values, row by row: the value at (row, column) is data[row * columns + column]. */
  readonly data: Float32Array;

  /**
   * Creates a rows x columns matrix.
   *
   * @param rows - The number of rows.
   * @param columns - The number of columns.
   * @param data - Optional values, row by row. Must be rows * columns long. When omitted, every value is 0.
   * @throws RangeError if data is given but is not rows * columns long.
   */
  constructor(rows: number, columns: number, data?: Float32Array) {
    if (data !== undefined && data.length !== rows * columns) {
      throw new RangeError(`Matrix| Expected ${rows * columns} values for a ${rows} x ${columns} matrix; got ${data.length}`);
    }
    this.rows = rows;
    this.columns = columns;
    this.data = data ?? new Float32Array(rows * columns);
  }

  /**
   * Creates a matrix from nested arrays, one inner array per row.
   *
   * @param values - The rows. Every row must be the same length.
   * @returns The new matrix.
   * @throws RangeError if the rows are not all the same length.
   */
  static fromArrays(values: number[][]): Matrix {
    const rows: number = values.length;
    const columns: number = rows > 0 ? values[0].length : 0;
    const matrix = new Matrix(rows, columns);
    for (let row = 0; row < rows; row++) {
      if (values[row].length !== columns) {
        throw new RangeError(`Matrix| Row ${row} has ${values[row].length} values; expected ${columns}`);
      }
      // Copy the row into its place in data: row r starts at index r * columns.
      matrix.data.set(values[row], row * columns);
    }
    return matrix;
  }

  /** The total number of values in the matrix: rows * columns. */
  get size(): number {
    return this.data.length;
  }

  /**
   * Reads one value.
   *
   * @param row - The row index, from 0 to rows - 1.
   * @param column - The column index, from 0 to columns - 1.
   * @returns The value at (row, column).
   */
  get(row: number, column: number): number {
    return this.data[row * this.columns + column];
  }

  /**
   * Changes one value.
   *
   * @param row - The row index, from 0 to rows - 1.
   * @param column - The column index, from 0 to columns - 1.
   * @param value - The new value. It is rounded to the nearest 32-bit float.
   */
  set(row: number, column: number, value: number): void {
    this.data[row * this.columns + column] = value;
  }

  /**
   * Returns one row. The result is a view onto the matrix's own storage, not
   * a copy: it is fast to create, but changing it changes the matrix.
   *
   * @param row - The row index, from 0 to rows - 1.
   * @returns The row's columns values.
   */
  getRow(row: number): Float32Array {
    return this.data.subarray(row * this.columns, (row + 1) * this.columns);
  }

  /**
   * Copies the matrix into nested arrays, one inner array per row. Useful for
   * printing with console.table, which lays nested arrays out as a grid.
   *
   * @returns The rows, as plain number arrays.
   */
  toArrays(): number[][] {
    return Array.from({ length: this.rows }, (_, row: number) => Array.from(this.getRow(row)));
  }
}

/**
 * Multiplies a vector by a matrix and optionally adds a bias: the
 * calculation done by a linear layer. If the vector has n values and the
 * matrix is n x m, the result has m values:
 *
 *     result[column] = biases[column] + sum over row of vector[row] * matrix(row, column)
 *
 * The loop goes through the matrix row by row, in the same order the values
 * are stored, which is much faster than going down each column in turn.
 * Each row's contribution, vector[row] times that row, is added to every
 * result value at once.
 *
 * @param vector - The input: matrix.rows values, e.g. one token's embedding.
 * @param matrix - The weights: one row per input value, one column per output value.
 * @param biases - Optional: matrix.columns values added to the result. Omit (or pass an empty array) for a layer with no bias.
 * @returns The output: matrix.columns values.
 * @throws RangeError if vector does not have matrix.rows values, or biases is not empty and does not have matrix.columns values.
 */
export function vectorTimesMatrix(vector: ArrayLike<number>, matrix: Matrix, biases?: ArrayLike<number>): number[] {
  if (vector.length !== matrix.rows) {
    throw new RangeError(`Matrix| Vector has ${vector.length} values; the ${matrix.rows} x ${matrix.columns} matrix needs ${matrix.rows}`);
  }
  const hasBiases: boolean = biases !== undefined && biases.length > 0;
  if (hasBiases && biases!.length !== matrix.columns) {
    throw new RangeError(`Matrix| Expected ${matrix.columns} biases; got ${biases!.length}`);
  }

  // Start from the biases, or from 0 when there are none.
  const result: number[] = [];
  for (let column = 0; column < matrix.columns; column++) {
    result.push(hasBiases ? biases![column] : 0);
  }

  for (let row = 0; row < matrix.rows; row++) {
    const value: number = vector[row];
    const rowStart: number = row * matrix.columns;
    for (let column = 0; column < matrix.columns; column++) {
      result[column] += value * matrix.data[rowStart + column];
    }
  }
  return result;
}

/**
 * Multiplies a matrix by a vector: each result value is the dot product of
 * one row of the matrix with the vector. If the matrix is m x n and the
 * vector has n values, the result has m values:
 *
 *     result[row] = sum over column of matrix(row, column) * vector[column]
 *
 * This is the same as vectorTimesMatrix with the transpose of the matrix,
 * without having to build the transpose. It is used where the weights are
 * stored with one row per output, such as the linear output layer, which
 * has one row per vocabulary token.
 *
 * @param matrix - The weights: one row per output value, one column per input value.
 * @param vector - The input: matrix.columns values.
 * @returns The output: matrix.rows values.
 * @throws RangeError if vector does not have matrix.columns values.
 */
export function matrixTimesVector(matrix: Matrix, vector: ArrayLike<number>): number[] {
  if (vector.length !== matrix.columns) {
    throw new RangeError(`Matrix| Vector has ${vector.length} values; the ${matrix.rows} x ${matrix.columns} matrix needs ${matrix.columns}`);
  }

  const result: number[] = new Array(matrix.rows);
  for (let row = 0; row < matrix.rows; row++) {
    const rowStart: number = row * matrix.columns;
    let sum = 0;
    for (let column = 0; column < matrix.columns; column++) {
      sum += matrix.data[rowStart + column] * vector[column];
    }
    result[row] = sum;
  }
  return result;
}

/*
 * Backpropagation through a linear layer
 * ---------------------------------------
 * A linear layer calculates output = input x weights (+ biases). Training
 * needs two things from each such layer:
 *
 * 1. The errors at its inputs, to pass back to the layer before. Each input
 *    value contributed to every output value through one row of weights, so
 *    its error is the dot product of that row with the output errors. This
 *    is the forward calculation with the weights transposed. The two
 *    multiplication functions above are each other's transpose, so no new
 *    function is needed:
 *
 *    | Weights stored                               | Forward                    | Backward (errors to inputs)      |
 *    |----------------------------------------------|----------------------------|----------------------------------|
 *    | input x output (GPTAttention,                | vectorTimesMatrix(x, W)    | matrixTimesVector(W, errors)     |
 *    |   MultiHeadAttention, FeedForward)           |                            |                                  |
 *    | output x input (LinearOutputLayer)           | matrixTimesVector(W, x)    | vectorTimesMatrix(errors, W)     |
 *
 * 2. The gradients of its weights and biases. Each weight connects one input
 *    to one output, so its gradient is that input times that output's error
 *    (addOuterProduct, below). Each bias is added straight to one output, so
 *    its gradient is that output's error (addToVector, below). The same
 *    weights are used for every token, so the gradients are summed over the
 *    tokens: these functions add to the gradients rather than replacing them.
 */

/**
 * Adds the outer product of two vectors to a gradient matrix: the weight
 * gradient of a linear layer for one token.
 *
 *     gradients(row, column) += input[row] * errors[column]
 *
 * The weight at (row, column) connects input row to output column, so the
 * rate at which the loss changes with that weight is the input value
 * multiplied by the output's error: dnn's "weight gradient = error x input".
 * Calling this once per token sums the gradient over every token.
 *
 * @param gradients - The gradient matrix to add to: input.length rows by errors.length columns, the same shape as the weights.
 * @param input - The layer's input for one token: one value per row.
 * @param errors - The errors at the layer's output for the same token: one value per column.
 * @throws RangeError if input does not have gradients.rows values or errors does not have gradients.columns values.
 */
export function addOuterProduct(gradients: Matrix, input: ArrayLike<number>, errors: ArrayLike<number>): void {
  if (input.length !== gradients.rows || errors.length !== gradients.columns) {
    throw new RangeError(
      `Matrix| A ${gradients.rows} x ${gradients.columns} gradient matrix needs ${gradients.rows} inputs and ` +
      `${gradients.columns} errors; got ${input.length} and ${errors.length}`
    );
  }

  // Row by row, in storage order: row r gets input[r] times the whole errors vector.
  for (let row = 0; row < gradients.rows; row++) {
    const value: number = input[row];
    const rowStart: number = row * gradients.columns;
    for (let column = 0; column < gradients.columns; column++) {
      gradients.data[rowStart + column] += value * errors[column];
    }
  }
}

/**
 * Adds one vector to another, element by element: used to sum bias
 * gradients over tokens, since a bias's gradient for one token is the error
 * of the output it is added to.
 *
 *     target[i] += values[i]
 *
 * @param target - The vector to add to, e.g. a bias gradient. Changed in place.
 * @param values - The values to add, e.g. one token's output errors. Must be the same length as target.
 * @throws RangeError if values is not the same length as target.
 */
export function addToVector(target: Float32Array, values: ArrayLike<number>): void {
  if (values.length !== target.length) {
    throw new RangeError(`Matrix| Cannot add ${values.length} values to a vector of ${target.length}`);
  }
  for (let i = 0; i < target.length; i++) {
    target[i] += values[i];
  }
}

export default Matrix;
