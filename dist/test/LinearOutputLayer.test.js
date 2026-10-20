import { test } from "node:test";
import assert from "node:assert/strict";
import Hyperparameters from "../Hyperparameters.js";
import LinearOutputLayer from "../LinearOutputLayer.js";
import { checkGradients, randomMatrix } from "./GradientCheck.js";
/*
 * Builds a LinearOutputLayer with a small embeddingSize (8) so the tests run
 * quickly. The vocabulary size always comes from the tokenizer (50,257), so
 * the gradient check samples the weights rather than checking all of them.
 */
const buildLayer = () => {
    const hyperparameters = new Hyperparameters(8, 4, 0, Hyperparameters.TRAINING_MODE, 2, 1);
    hyperparameters.seed = 11;
    const layer = new LinearOutputLayer(hyperparameters);
    layer.build();
    return layer;
};
test("backpropagation gives the same weight gradients and input errors as the numerical gradients", () => {
    const layer = buildLayer();
    const inputs = randomMatrix(4, 8, 1);
    let inputErrors = [];
    checkGradients({
        forward: () => layer.update(inputs),
        backward: (outputErrors) => {
            inputErrors = layer.backpropagate(outputErrors);
            layer.calculateGradients(false);
        },
        parameters: [{ name: "weights", values: layer.weights.data, gradients: () => layer.weightGradients.data }],
        inputs,
        inputErrors: () => inputErrors,
    });
});
test("calculateGradients(true) averages windows: two windows give the mean of their separate gradients", () => {
    const layer = buildLayer();
    const inputsA = randomMatrix(4, 8, 2);
    const inputsB = randomMatrix(4, 8, 3);
    const errorsA = randomMatrix(4, layer.vocabSize, 4);
    const errorsB = randomMatrix(4, layer.vocabSize, 5);
    // The gradients of each window on its own.
    const gradientsOf = (inputs, errors) => {
        layer.update(inputs);
        layer.backpropagate(errors);
        layer.calculateGradients(false);
        return Float32Array.from(layer.weightGradients.data);
    };
    const gradientsA = gradientsOf(inputsA, errorsA);
    const gradientsB = gradientsOf(inputsB, errorsB);
    // Window A starts the batch; window B is averaged in.
    layer.update(inputsA);
    layer.backpropagate(errorsA);
    layer.calculateGradients(false);
    assert.equal(layer.gradientCount, 1);
    layer.update(inputsB);
    layer.backpropagate(errorsB);
    layer.calculateGradients(true);
    assert.equal(layer.gradientCount, 2);
    const averaged = layer.weightGradients.data;
    for (let i = 0; i < averaged.length; i += 997) {
        const expected = (gradientsA[i] + gradientsB[i]) / 2;
        assert.ok(Math.abs(averaged[i] - expected) <= 1e-6 * Math.max(1, Math.abs(expected)), `gradient ${i}`);
    }
    // A new batch starts again from a count of 1.
    layer.calculateGradients(false);
    assert.equal(layer.gradientCount, 1);
});
test("backpropagate() rejects errors for a different number of tokens", () => {
    const layer = buildLayer();
    layer.update(randomMatrix(4, 8, 1));
    assert.throws(() => layer.backpropagate(randomMatrix(3, layer.vocabSize, 1)), RangeError);
});
