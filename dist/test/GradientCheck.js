import assert from "node:assert/strict";
import createSeededRandom from "../SeededRandom.js";
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
export function checkGradients(options) {
    const parameters = options.parameters ?? [];
    const samplesPerArray = options.samplesPerArray ?? 20;
    const step = options.step ?? 1e-2;
    const relativeTolerance = options.relativeTolerance ?? 1e-2;
    const absoluteTolerance = options.absoluteTolerance ?? 1e-5;
    const random = createSeededRandom(options.seed ?? 1);
    // Step 1: the backward pass, with R as the output errors.
    const output = options.forward();
    const outputErrors = output.map((row) => row.map(() => 2 * random() - 1));
    options.backward(outputErrors);
    // The loss whose gradients are being checked: L = sum of output * R.
    const loss = () => {
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
    const analyticParameterGradients = parameters.map((parameter) => Array.from(parameter.gradients()));
    const checkInputs = options.inputs !== undefined && options.inputErrors !== undefined;
    const analyticInputErrors = checkInputs ? options.inputErrors().map((row) => [...row]) : [];
    /**
     * Chooses which indices of an array to check: all of them if the array
     * is small, otherwise samplesPerArray distinct indices chosen at random.
     */
    const chooseIndices = (length) => {
        if (length <= samplesPerArray) {
            return Array.from({ length }, (_, i) => i);
        }
        const chosen = new Set();
        while (chosen.size < samplesPerArray) {
            chosen.add(Math.floor(random() * length));
        }
        return [...chosen];
    };
    /** True if the analytic and numerical gradients agree within the tolerances. */
    const agrees = (analytic, numerical) => Math.abs(analytic - numerical) <= absoluteTolerance + relativeTolerance * Math.max(Math.abs(analytic), Math.abs(numerical));
    const mismatches = [];
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
        const inputs = options.inputs;
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
