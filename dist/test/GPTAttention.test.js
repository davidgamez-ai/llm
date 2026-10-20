import { test } from "node:test";
import assert from "node:assert/strict";
import Hyperparameters from "../Hyperparameters.js";
import GPTAttention from "../GPTAttention.js";
import { checkGradients, randomMatrix } from "./GradientCheck.js";
/*
 * Builds a GPTAttention head over embeddings of 8 values, with 2 heads'
 * worth of columns (4). The weights use a larger standard deviation (0.5)
 * than GPT-2's 0.02, so the attention weights are far from uniform and the
 * softmax is properly exercised. Biases, when used, are set to random values.
 */
const setUp = (qkvBias, dropoutRate = 0) => {
    const hyperparameters = new Hyperparameters(8, 4, dropoutRate, Hyperparameters.TRAINING_MODE, 2, 1);
    hyperparameters.qkvBias = qkvBias;
    hyperparameters.linearLayerStandardDeviation = 0.5;
    hyperparameters.seed = 41;
    const head = new GPTAttention(hyperparameters);
    if (qkvBias) {
        head.queryBiases.set(randomMatrix(1, 4, 42, 0.5)[0]);
        head.keyBiases.set(randomMatrix(1, 4, 43, 0.5)[0]);
        head.valueBiases.set(randomMatrix(1, 4, 44, 0.5)[0]);
    }
    return { hyperparameters, head };
};
/**
 * Runs the gradient check on a head, for its weights, its biases (when it
 * has them) and its inputs.
 *
 * @param head - The head to check.
 * @param beforeForward - Called before every forward pass, e.g. to reset the random seed so dropout draws the same mask each time.
 */
const check = (head, beforeForward = () => { }) => {
    const inputs = randomMatrix(4, 8, 45, 1.5);
    let inputErrors = [];
    const parameters = [
        { name: "queryWeights", values: head.queryWeights.data, gradients: () => head.queryWeightGradients.data },
        { name: "keyWeights", values: head.keyWeights.data, gradients: () => head.keyWeightGradients.data },
        { name: "valueWeights", values: head.valueWeights.data, gradients: () => head.valueWeightGradients.data },
    ];
    if (head.qkvBias) {
        parameters.push({ name: "queryBiases", values: head.queryBiases, gradients: () => head.queryBiasGradients }, { name: "keyBiases", values: head.keyBiases, gradients: () => head.keyBiasGradients }, { name: "valueBiases", values: head.valueBiases, gradients: () => head.valueBiasGradients });
    }
    checkGradients({
        forward: () => {
            beforeForward();
            return head.update(inputs);
        },
        backward: (outputErrors) => {
            inputErrors = head.backpropagate(outputErrors);
            head.calculateGradients(false);
        },
        parameters,
        inputs,
        inputErrors: () => inputErrors,
    });
};
test("backpropagation without biases gives the same gradients and input errors as the numerical gradients", () => {
    check(setUp(false).head);
});
test("backpropagation with biases gives the same gradients and input errors as the numerical gradients", () => {
    check(setUp(true).head);
});
test("backpropagation through dropout on the attention weights matches the numerical gradients", () => {
    const { hyperparameters, head } = setUp(false, 0.5);
    // Resetting the seed before every forward pass makes dropout draw the same mask each time.
    check(head, () => { hyperparameters.seed = 46; });
    assert.ok(head.attentionDropout.mask.some((row) => row.includes(0)), "some attention weights were dropped");
});
test("because of the causal mask, errors at token 0's context vector reach only token 0's input", () => {
    const { head } = setUp(false);
    head.update(randomMatrix(4, 8, 47));
    // Errors only on token 0's context vector. Token 0 attends only to itself, so no other token can be affected.
    const outputErrors = [[1, -1, 0.5, 2], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]];
    const inputErrors = head.backpropagate(outputErrors);
    assert.ok(inputErrors[0].some((error) => error !== 0));
    for (let token = 1; token < 4; token++) {
        assert.ok(inputErrors[token].every((error) => error === 0), `token ${token} received errors`);
    }
});
test("backpropagate() rejects errors for a different number of tokens", () => {
    const { head } = setUp(false);
    head.update(randomMatrix(4, 8, 48));
    assert.throws(() => head.backpropagate(randomMatrix(3, 4, 1)), RangeError);
});
