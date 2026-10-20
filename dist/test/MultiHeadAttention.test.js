import { test } from "node:test";
import assert from "node:assert/strict";
import Hyperparameters from "../Hyperparameters.js";
import MultiHeadAttention from "../MultiHeadAttention.js";
import { checkGradients, randomMatrix } from "./GradientCheck.js";
/*
 * Builds a MultiHeadAttention with 2 heads over embeddings of 8 values. The
 * weights use a larger standard deviation (0.5) than GPT-2's 0.02, so the
 * attention weights are far from uniform, and the output projection's
 * biases are set to random values.
 */
const buildAttention = () => {
    const hyperparameters = new Hyperparameters(8, 4, 0, Hyperparameters.TRAINING_MODE, 2, 1);
    hyperparameters.linearLayerStandardDeviation = 0.5;
    hyperparameters.seed = 51;
    const attention = new MultiHeadAttention(hyperparameters);
    attention.outputProjectionBiases.set(randomMatrix(1, 8, 52, 0.5)[0]);
    return attention;
};
test("backpropagation gives the same gradients, including each head's, and input errors as the numerical gradients", () => {
    const attention = buildAttention();
    const [head0, head1] = attention.attentionHeads;
    const inputs = randomMatrix(4, 8, 53, 1.5);
    let inputErrors = [];
    checkGradients({
        forward: () => attention.update(inputs),
        backward: (outputErrors) => {
            inputErrors = attention.backpropagate(outputErrors);
            attention.calculateGradients(false);
        },
        parameters: [
            { name: "outputProjectionWeights", values: attention.outputProjectionWeights.data, gradients: () => attention.outputProjectionWeightGradients.data },
            { name: "outputProjectionBiases", values: attention.outputProjectionBiases, gradients: () => attention.outputProjectionBiasGradients },
            // One weight matrix from each head, to check the errors are split between the heads correctly.
            { name: "head 0 queryWeights", values: head0.queryWeights.data, gradients: () => head0.queryWeightGradients.data },
            { name: "head 1 keyWeights", values: head1.keyWeights.data, gradients: () => head1.keyWeightGradients.data },
            { name: "head 1 valueWeights", values: head1.valueWeights.data, gradients: () => head1.valueWeightGradients.data },
        ],
        inputs,
        inputErrors: () => inputErrors,
    });
});
test("calculateGradients() also calculates every head's gradients", () => {
    const attention = buildAttention();
    attention.update(randomMatrix(4, 8, 54));
    attention.backpropagate(randomMatrix(4, 8, 55));
    attention.calculateGradients(false);
    for (const head of attention.attentionHeads) {
        assert.equal(head.gradientCount, 1);
        assert.equal(head.valueWeightGradients.size, head.valueWeights.size);
    }
});
test("backpropagate() rejects errors for a different number of tokens", () => {
    const attention = buildAttention();
    attention.update(randomMatrix(4, 8, 56));
    assert.throws(() => attention.backpropagate(randomMatrix(3, 8, 1)), RangeError);
});
