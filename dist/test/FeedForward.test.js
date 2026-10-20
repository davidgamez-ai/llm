import { test } from "node:test";
import assert from "node:assert/strict";
import Hyperparameters from "../Hyperparameters.js";
import FeedForward, { gelu, geluDerivative } from "../FeedForward.js";
import { checkGradients, randomMatrix } from "./GradientCheck.js";
/*
 * Builds a FeedForward with embeddingSize 8 (so a hidden size of 32). The
 * biases are set to random values, rather than their initial 0s, so the
 * tests cover the general case.
 */
const buildNetwork = (bias) => {
    const hyperparameters = new Hyperparameters(8, 4, 0, Hyperparameters.TRAINING_MODE, 2, 1);
    hyperparameters.feedForwardBias = bias;
    hyperparameters.linearLayerStandardDeviation = 0.3;
    hyperparameters.seed = 31;
    const network = new FeedForward(hyperparameters);
    network.firstLayerBiases.set(randomMatrix(1, network.firstLayerBiases.length, 32, 0.5)[0]);
    network.secondLayerBiases.set(randomMatrix(1, network.secondLayerBiases.length, 33, 0.5)[0]);
    return network;
};
/**
 * Runs the gradient check on a network, for its weights, its biases (when
 * it has them) and its inputs.
 */
const check = (network) => {
    const inputs = randomMatrix(4, 8, 34, 2);
    let inputErrors = [];
    const parameters = [
        { name: "firstLayerWeights", values: network.firstLayerWeights.data, gradients: () => network.firstLayerWeightGradients.data },
        { name: "secondLayerWeights", values: network.secondLayerWeights.data, gradients: () => network.secondLayerWeightGradients.data },
    ];
    if (network.bias) {
        parameters.push({ name: "firstLayerBiases", values: network.firstLayerBiases, gradients: () => network.firstLayerBiasGradients }, { name: "secondLayerBiases", values: network.secondLayerBiases, gradients: () => network.secondLayerBiasGradients });
    }
    checkGradients({
        forward: () => network.update(inputs),
        backward: (outputErrors) => {
            inputErrors = network.backpropagate(outputErrors);
            network.calculateGradients(false);
        },
        parameters,
        inputs,
        inputErrors: () => inputErrors,
    });
};
test("geluDerivative() matches the numerical slope of gelu() from -4 to 4", () => {
    const h = 1e-5;
    for (let x = -4; x <= 4; x += 0.25) {
        const numerical = (gelu(x + h) - gelu(x - h)) / (2 * h);
        assert.ok(Math.abs(geluDerivative(x) - numerical) < 1e-8, `x = ${x}: ${geluDerivative(x)} vs ${numerical}`);
    }
});
test("gelu() passes large positive values through, removes large negative ones, and gelu(0) = 0", () => {
    assert.equal(gelu(0), 0);
    assert.ok(Math.abs(gelu(10) - 10) < 1e-9);
    assert.ok(Math.abs(gelu(-10)) < 1e-9);
});
test("backpropagation with biases gives the same gradients and input errors as the numerical gradients", () => {
    check(buildNetwork(true));
});
test("backpropagation without biases gives the same gradients and input errors, and leaves the bias gradients empty", () => {
    const network = buildNetwork(false);
    check(network);
    assert.equal(network.firstLayerBiasGradients.length, 0);
    assert.equal(network.secondLayerBiasGradients.length, 0);
});
