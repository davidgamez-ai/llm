import Hyperparameters from "./Hyperparameters.js";
import Matrix from "./Matrix.js";

/**
 * The running averages AdamW keeps for one parameter array, one value per
 * parameter.
 */
interface Moments {
  /** The running average of the gradients (the "first moment"): which way the parameter has been pushed recently. */
  m: Float32Array;

  /** The running average of the squared gradients (the "second moment"): how large its gradients have been recently. */
  v: Float32Array;
}

/**
 * The AdamW optimizer, which uses the gradients calculated by
 * backpropagation to update the model's parameters. A port of dnn's AdamW,
 * adapted to the Float32Array storage used by this model.
 *
 * Plain gradient descent moves every parameter by learningRate times its
 * gradient. AdamW improves on this in two ways:
 *
 * - Momentum: it moves each parameter according to a running average of
 *   its recent gradients (m) rather than the latest one alone, so the noise
 *   in individual batches partly cancels out.
 * - A separate step size for every parameter: it divides by the square root
 *   of a running average of the squared gradients (v), so parameters whose
 *   gradients are always large take smaller steps, and those whose gradients
 *   are always small take larger ones. Each step moves a parameter by
 *   roughly learningRate, whatever the size of its gradients.
 *
 * The "W" is decoupled weight decay: each step also shrinks the parameters
 * a little towards 0, separately from the gradient step, which discourages
 * large weights and so reduces overfitting.
 *
 * One AdamW instance is shared by the whole model. Call step() once per
 * batch, then updateMatrix() or updateVector() once for every parameter
 * array. The hyperparameters are read from Hyperparameters on every update,
 * so changing them (e.g. the learning rate) takes effect immediately.
 */
class AdamW {
  /** Source of learningRate, beta1, beta2, optimizerEpsilon and weightDecay. */
  private readonly hyperparameters: Hyperparameters;

  /** The number of times step() has been called, i.e. the number of batches so far. */
  private _stepCount: number = 0;

  /**
   * The running averages of each parameter array, keyed by the array
   * itself, so each array has its own. A WeakMap lets an array's moments be
   * garbage collected if the array is no longer used (e.g. if a layer is
   * rebuilt).
   */
  private readonly moments: WeakMap<Float32Array, Moments> = new WeakMap();

  /**
   * Creates an AdamW optimizer with no steps taken.
   *
   * @param hyperparameters - Source of learningRate, beta1, beta2, optimizerEpsilon and weightDecay, read on every update.
   */
  constructor(hyperparameters: Hyperparameters) {
    this.hyperparameters = hyperparameters;
  }

  /** The number of times step() has been called. */
  get stepCount(): number {
    return this._stepCount;
  }

  /**
   * Starts a new optimization step. Must be called once per batch, before
   * any parameters are updated. The step count is used to correct the
   * running averages for starting at zero (see updateVector).
   */
  step(): void {
    this._stepCount++;
  }

  /**
   * Updates a weight matrix from its gradients. A Matrix stores its values
   * in a single Float32Array, so this simply updates that array.
   *
   * @param weights - The weights to update, changed in place.
   * @param gradients - Their gradients: the same shape as weights.
   * @param applyWeightDecay - True (the default) to shrink the weights towards 0 as well.
   * @throws Error if step() has not been called.
   * @throws RangeError if gradients is not the same shape as weights.
   */
  updateMatrix(weights: Matrix, gradients: Matrix, applyWeightDecay: boolean = true): void {
    if (gradients.rows !== weights.rows || gradients.columns !== weights.columns) {
      throw new RangeError(
        `AdamW| Gradients are ${gradients.rows} x ${gradients.columns}; the weights are ${weights.rows} x ${weights.columns}`
      );
    }
    this.updateVector(weights.data, gradients.data, applyWeightDecay);
  }

  /**
   * Updates an array of parameters from their gradients. For each parameter p
   * with gradient g, at step t:
   *
   *     m = beta1 * m + (1 - beta1) * g          running average of the gradients
   *     v = beta2 * v + (1 - beta2) * g^2        running average of the squared gradients
   *     mHat = m / (1 - beta1^t)                 corrected for m and v starting at 0
   *     vHat = v / (1 - beta2^t)
   *     p = p - learningRate * weightDecay * p   decoupled weight decay (if applied)
   *     p = p - learningRate * mHat / (sqrt(vHat) + epsilon)
   *
   * Why the correction? m and v start at 0, so for the first few steps they
   * are much smaller than the gradients they are averaging: after one step,
   * m is only (1 - beta1) * g, a tenth of g. Dividing by (1 - beta1^t) scales
   * them back up; as t grows, beta1^t shrinks towards 0 and the correction
   * fades away. On the first step mHat = g and vHat = g^2, so every
   * parameter moves by almost exactly learningRate.
   *
   * @param parameters - The parameters to update, e.g. biases, changed in place.
   * @param gradients - Their gradients: the same length as parameters.
   * @param applyWeightDecay - True to shrink the parameters towards 0 as well. Defaults to false, since this is mostly used for biases and layer normalization, which are not decayed.
   * @throws Error if step() has not been called.
   * @throws RangeError if gradients is not the same length as parameters.
   */
  updateVector(parameters: Float32Array, gradients: ArrayLike<number>, applyWeightDecay: boolean = false): void {
    if (this._stepCount === 0) {
      throw new Error("AdamW| Call step() before updating parameters.");
    }
    if (gradients.length !== parameters.length) {
      throw new RangeError(`AdamW| Got ${gradients.length} gradients for ${parameters.length} parameters`);
    }

    // The first time an array is updated, its running averages start at 0.
    let moments = this.moments.get(parameters);
    if (moments === undefined) {
      moments = { m: new Float32Array(parameters.length), v: new Float32Array(parameters.length) };
      this.moments.set(parameters, moments);
    }
    const { m, v } = moments;

    const { learningRate, beta1, beta2, optimizerEpsilon, weightDecay } = this.hyperparameters;
    const t = this._stepCount;
    const firstMomentCorrection = 1 - beta1 ** t;
    const secondMomentCorrection = 1 - beta2 ** t;

    for (let i = 0; i < parameters.length; i++) {
      const gradient = gradients[i];

      // Update the running averages of the gradient and the squared gradient.
      m[i] = beta1 * m[i] + (1 - beta1) * gradient;
      v[i] = beta2 * v[i] + (1 - beta2) * gradient * gradient;

      // Correct them for starting at 0.
      const mHat = m[i] / firstMomentCorrection;
      const vHat = v[i] / secondMomentCorrection;

      // Weight decay shrinks the parameter towards 0, separately from the gradient.
      let parameter = parameters[i];
      if (applyWeightDecay) {
        parameter -= learningRate * weightDecay * parameter;
      }

      // Move against the averaged gradient, scaled by its typical size.
      parameters[i] = parameter - (learningRate * mHat) / (Math.sqrt(vHat) + optimizerEpsilon);
    }
  }
}

export default AdamW;
