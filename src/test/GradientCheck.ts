import assert from "node:assert/strict";
import createSeededRandom from "../SeededRandom.js";

/*
 * Gradient checking: a test helper that compares the gradients calculated
 * by hand-written backpropagation with gradients measured numerically.
 *
 * The numerical gradient of a loss L with respect to a value v is found by
 * nudging v up and down by a small step h and measuring the change in L:
 *
 *     dL/dv ≈ (L(v + h) - L(v - h)) / 2h
 *
 * This needs no knowledge of how the component works, only its forward
 * pass, so it is an independent check of the backward pass. It is far too
 * slow for training (two forward passes per value), but ideal for tests.
 *
 * Any component can be checked, because the loss is made up for the test:
 *
 *     L = sum over every output value of output * R
 *
 * for a fixed random matrix R. The derivative of L with respect to each
 * output value is then just R, so R is the "output errors" passed to the
 * component's backpropagation.
 */

/** A trainable parameter array to check, with the gradients calculated for it by backpropagation. */
export interface CheckedParameter {
  /** Name of the array, used in failure messages, e.g. "queryWeights". */
  name: string;

  /** The parameter values themselves (e.g. matrix.data or a bias array). Nudged in place during the check, then restored. */
  values: Float32Array;

  /** Returns the gradients calculated by backpropagation, the same length as values. Called after backward(). */
  gradients: () => ArrayLike<number>;
}

/** What checkGradients needs to know about the component being checked. */
export interface GradientCheckOptions {
  /**
   * Runs the component's forward pass and returns its output. Must read the
   * current contents of inputs and parameters (they are changed in place
   * between calls), and must be deterministic: use a dropout rate of 0, or
   * reset the random seed at the start of every call.
   */
  forward: () => number[][];

  /**
   * Runs the component's backward pass with the given errors at its output:
   * typically backpropagate(outputErrors) followed by calculateGradients(false).
   */
  backward: (outputErrors: number[][]) => void;

  /** The parameter arrays whose gradients are checked. */
  parameters?: CheckedParameter[];

  /** The input to the component, read by forward(). If given with inputErrors, the errors at the inputs are checked too. */
  inputs?: number[][];

  /** Returns the errors at the inputs calculated by backpropagation, the same shape as inputs. Called after backward(). */
  inputErrors?: () => number[][];

  /** How many values to check in each parameter array and in the inputs. Arrays this size or smaller are checked in full. Defaults to 20. */
  samplesPerArray?: number;

  /** The step h by which each value is nudged. Defaults to 1e-2: parameters are stored as 32-bit floats, so a smaller step is swamped by rounding. */
  step?: number;

  /** The largest allowed difference, relative to the larger of the two gradients. Defaults to 1e-2. */
  relativeTolerance?: number;

  /** A difference this small always passes, so gradients that should be 0 are not failed by rounding. Defaults to 1e-5. */
  absoluteTolerance?: number;

  /** Seed for R and for choosing which values to check. Defaults to 1. */
  seed?: number;
}

/** One value whose analytic and numerical gradients disagreed. */
interface Mismatch {
  name: string;
  index: number;
  analytic: number;
  numerical: number;
}

/**
 * Checks a component's hand-written gradients against numerical ones:
 *
 * 1. Runs forward(), builds a random matrix R the shape of the output, and
 *    runs backward(R). The analytic gradients are copied, because later
 *    forward passes overwrite the values the component stores.
 * 2. For a sample of values in each parameter array, and in the inputs,
 *    nudges the value up and down by step, runs forward() each time to
 *    measure the loss L = sum of output * R, and restores the value.
 * 3. Compares each numerical gradient with the analytic one.
 *
 * @param options - The component's forward and backward passes, and what to check.
 * @returns The number of values checked.
 * @throws AssertionError listing the values whose gradients disagree.
 */
export function checkGradients(options: GradientCheckOptions): number {
  const parameters = options.parameters ?? [];
  const samplesPerArray = options.samplesPerArray ?? 20;
  const step = options.step ?? 1e-2;
  const relativeTolerance = options.relativeTolerance ?? 1e-2;
  const absoluteTolerance = options.absoluteTolerance ?? 1e-5;
  const random = createSeededRandom(options.seed ?? 1);

  // Step 1: the backward pass, with R as the output errors.
  const output = options.forward();
  const outputErrors: number[][] = output.map((row) => row.map(() => 2 * random() - 1));
  options.backward(outputErrors);

  // The loss whose gradients are being checked: L = sum of output * R.
  const loss = (): number => {
    const result = options.forward();
    let total = 0;
    for (let i = 0; i < result.length; i++) {
      for (let j = 0; j < result[i].length; j++) {
        total += result[i][j] * outputErrors[i][j];
      }
    }
    return total;
  };

  // Copy the analytic gradients before any further forward passes.
  const analyticParameterGradients: number[][] = parameters.map((parameter) => Array.from(parameter.gradients()));
  const checkInputs: boolean = options.inputs !== undefined && options.inputErrors !== undefined;
  const analyticInputErrors: number[][] = checkInputs ? options.inputErrors!().map((row) => [...row]) : [];

  /**
   * Chooses which indices of an array to check: all of them if the array
   * is small, otherwise samplesPerArray distinct indices chosen at random.
   */
  const chooseIndices = (length: number): number[] => {
    if (length <= samplesPerArray) {
      return Array.from({ length }, (_, i) => i);
    }
    const chosen = new Set<number>();
    while (chosen.size < samplesPerArray) {
      chosen.add(Math.floor(random() * length));
    }
    return [...chosen];
  };

  /** True if the analytic and numerical gradients agree within the tolerances. */
  const agrees = (analytic: number, numerical: number): boolean =>
    Math.abs(analytic - numerical) <= absoluteTolerance + relativeTolerance * Math.max(Math.abs(analytic), Math.abs(numerical));

  const mismatches: Mismatch[] = [];
  let checked = 0;

  // Step 2 and 3 for the parameters.
  parameters.forEach((parameter, p) => {
    for (const index of chooseIndices(parameter.values.length)) {
      const original = parameter.values[index];

      parameter.values[index] = original + step;
      // Read the value back: 32-bit storage rounds original + step, and the true change must be used.
      const plusValue = parameter.values[index];
      const plusLoss = loss();

      parameter.values[index] = original - step;
      const minusValue = parameter.values[index];
      const minusLoss = loss();

      // original came from the Float32Array, so restoring it is exact.
      parameter.values[index] = original;

      const numerical = (plusLoss - minusLoss) / (plusValue - minusValue);
      const analytic = analyticParameterGradients[p][index];
      checked++;
      if (!agrees(analytic, numerical)) {
        mismatches.push({ name: parameter.name, index, analytic, numerical });
      }
    }
  });

  // Steps 2 and 3 for the inputs, which are ordinary 64-bit numbers.
  if (checkInputs) {
    const inputs = options.inputs!;
    const width = inputs[0]?.length ?? 0;
    for (const index of chooseIndices(inputs.length * width)) {
      const row = Math.floor(index / width);
      const column = index % width;
      const original = inputs[row][column];

      inputs[row][column] = original + step;
      const plusLoss = loss();
      inputs[row][column] = original - step;
      const minusLoss = loss();
      inputs[row][column] = original;

      const numerical = (plusLoss - minusLoss) / (2 * step);
      const analytic = analyticInputErrors[row][column];
      checked++;
      if (!agrees(analytic, numerical)) {
        mismatches.push({ name: `inputs[${row}]`, index: column, analytic, numerical });
      }
    }
  }

  if (mismatches.length > 0) {
    const details = mismatches
      .slice(0, 10)
      .map((m) => `  ${m.name}[${m.index}]: analytic ${m.analytic}, numerical ${m.numerical}`)
      .join("\n");
    assert.fail(`GradientCheck| ${mismatches.length} of ${checked} gradients disagree:\n${details}`);
  }

  return checked;
}
