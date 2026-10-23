import { test } from "node:test";
import assert from "node:assert/strict";
import Hyperparameters from "../Hyperparameters.js";
import Decoder from "../Decoder.js";
/** The number of tokens drawn when checking how often each token is chosen. */
const DRAWS = 3000;
/**
 * Creates seeded generation-mode Hyperparameters with the given sampling
 * method and temperature.
 *
 * @param samplingMethod - GREEDY_SAMPLING, TOP_K_SAMPLING or TOP_P_SAMPLING.
 * @param temperature - The temperature to sample with. Defaults to 1.
 */
const generationHyperparameters = (samplingMethod, temperature = 1) => {
    const hyperparameters = new Hyperparameters();
    hyperparameters.mode = Hyperparameters.GENERATION_MODE;
    hyperparameters.samplingMethod = samplingMethod;
    hyperparameters.temperature = temperature;
    hyperparameters.seed = 42;
    return hyperparameters;
};
/**
 * Creates vocabularySize logits that are all -Infinity (probability 0)
 * except for the given token ids and values.
 *
 * @param hyperparameters - Source of vocabularySize.
 * @param values - Logits by token id; every other token gets -Infinity.
 */
const logitsWith = (hyperparameters, values) => {
    const logits = new Array(hyperparameters.vocabularySize).fill(-Infinity);
    for (const [tokenId, value] of Object.entries(values)) {
        logits[Number(tokenId)] = value;
    }
    return logits;
};
/** Decodes DRAWS tokens and returns how often each token id was chosen. */
const decodeCounts = (decoder, logits) => {
    const counts = new Map();
    for (let i = 0; i < DRAWS; i++) {
        const tokenId = decoder.decode(logits);
        counts.set(tokenId, (counts.get(tokenId) ?? 0) + 1);
    }
    return counts;
};
/** The chosen token ids, in increasing order. */
const chosenIds = (counts) => [...counts.keys()].sort((a, b) => a - b);
/** The share of the draws that chose the given token. */
const share = (counts, tokenId) => (counts.get(tokenId) ?? 0) / DRAWS;
test("decode() uses greedy decoding by default, whatever the temperature", () => {
    const hyperparameters = generationHyperparameters(Hyperparameters.GREEDY_SAMPLING);
    const logits = logitsWith(hyperparameters, { 3: 1, 7: 1.5, 9: 1.4 });
    assert.deepEqual(chosenIds(decodeCounts(new Decoder(hyperparameters), logits)), [7]);
});
test("top-k and top-p at temperature 0 always pick the highest logit", () => {
    for (const samplingMethod of [Hyperparameters.TOP_K_SAMPLING, Hyperparameters.TOP_P_SAMPLING]) {
        const hyperparameters = generationHyperparameters(samplingMethod, 0);
        const logits = logitsWith(hyperparameters, { 3: 1, 7: 1.5, 9: 1.4 });
        assert.deepEqual(chosenIds(decodeCounts(new Decoder(hyperparameters), logits)), [7]);
    }
});
test("top-k only chooses between the k highest logits", () => {
    const hyperparameters = generationHyperparameters(Hyperparameters.TOP_K_SAMPLING);
    hyperparameters.topK = 2;
    const logits = logitsWith(hyperparameters, { 1: 3, 2: 2, 3: 1, 4: 0 });
    const counts = decodeCounts(new Decoder(hyperparameters), logits);
    assert.deepEqual(chosenIds(counts), [1, 2]);
    // The kept tokens keep their relative probabilities: e^3 to e^2.
    assert.ok(Math.abs(share(counts, 1) - Math.E / (Math.E + 1)) < 0.03, `token 1 chosen ${counts.get(1)} times`);
});
test("top-k keeps every token that ties with the k-th highest logit", () => {
    const hyperparameters = generationHyperparameters(Hyperparameters.TOP_K_SAMPLING);
    hyperparameters.topK = 2;
    const logits = logitsWith(hyperparameters, { 1: 2, 2: 1, 3: 1, 4: 0 });
    assert.deepEqual(chosenIds(decodeCounts(new Decoder(hyperparameters), logits)), [1, 2, 3]);
});
test("top-k at temperature 1 picks tokens in proportion to the softmax of the logits", () => {
    const hyperparameters = generationHyperparameters(Hyperparameters.TOP_K_SAMPLING);
    // Probabilities 1/3 and 2/3.
    const logits = logitsWith(hyperparameters, { 10: 0, 20: Math.log(2) });
    const counts = decodeCounts(new Decoder(hyperparameters), logits);
    assert.deepEqual(chosenIds(counts), [10, 20]);
    assert.ok(Math.abs(share(counts, 20) - 2 / 3) < 0.03, `token 20 chosen ${counts.get(20)} times`);
});
test("a lower temperature makes the most likely token more likely to be chosen", () => {
    const shareAt = (temperature) => {
        const hyperparameters = generationHyperparameters(Hyperparameters.TOP_K_SAMPLING, temperature);
        const logits = logitsWith(hyperparameters, { 10: 0, 20: Math.log(2) });
        return share(decodeCounts(new Decoder(hyperparameters), logits), 20);
    };
    // At temperature 0.5 the odds are squared: 4 to 1, so 0.8.
    assert.ok(Math.abs(shareAt(0.5) - 0.8) < 0.03);
    assert.ok(shareAt(0.1) > shareAt(0.5));
});
test("top-p keeps the most likely tokens until their probabilities reach topP", () => {
    const hyperparameters = generationHyperparameters(Hyperparameters.TOP_P_SAMPLING);
    // Probabilities 0.5, 0.3 and 0.2.
    const logits = logitsWith(hyperparameters, { 1: Math.log(0.5), 2: Math.log(0.3), 3: Math.log(0.2) });
    const decoder = new Decoder(hyperparameters);
    // 0.5 falls short of 0.7, and 0.5 + 0.3 reaches it, so tokens 1 and 2 are kept, in the ratio 5 to 3.
    hyperparameters.topP = 0.7;
    const counts = decodeCounts(decoder, logits);
    assert.deepEqual(chosenIds(counts), [1, 2]);
    assert.ok(Math.abs(share(counts, 1) - 0.625) < 0.03, `token 1 chosen ${counts.get(1)} times`);
    // The most likely token alone reaches 0.4.
    hyperparameters.topP = 0.4;
    assert.deepEqual(chosenIds(decodeCounts(decoder, logits)), [1]);
    // At 1 every token is kept.
    hyperparameters.topP = 1;
    assert.deepEqual(chosenIds(decodeCounts(decoder, logits)), [1, 2, 3]);
});
test("decode() reads the sampling method on every call", () => {
    const hyperparameters = generationHyperparameters(Hyperparameters.TOP_K_SAMPLING);
    const decoder = new Decoder(hyperparameters);
    const logits = logitsWith(hyperparameters, { 10: 0, 20: 0.1 });
    assert.deepEqual(chosenIds(decodeCounts(decoder, logits)), [10, 20]);
    hyperparameters.samplingMethod = Hyperparameters.GREEDY_SAMPLING;
    assert.deepEqual(chosenIds(decodeCounts(decoder, logits)), [20]);
});
test("decode() is reproducible for a given seed", () => {
    for (const samplingMethod of [Hyperparameters.TOP_K_SAMPLING, Hyperparameters.TOP_P_SAMPLING]) {
        const draw = () => {
            const hyperparameters = generationHyperparameters(samplingMethod);
            const decoder = new Decoder(hyperparameters);
            const logits = logitsWith(hyperparameters, { 1: 0, 2: 0, 3: 0, 4: 0 });
            return Array.from({ length: 20 }, () => decoder.decode(logits));
        };
        assert.deepEqual(draw(), draw());
    }
});
test("every sampling method rejects the wrong number of logits and modes other than generation", () => {
    for (const samplingMethod of [Hyperparameters.GREEDY_SAMPLING, Hyperparameters.TOP_K_SAMPLING, Hyperparameters.TOP_P_SAMPLING]) {
        const hyperparameters = generationHyperparameters(samplingMethod);
        const decoder = new Decoder(hyperparameters);
        assert.throws(() => decoder.decode([0, 1, 2]), RangeError);
        hyperparameters.mode = Hyperparameters.TRAINING_MODE;
        assert.throws(() => decoder.decode(logitsWith(hyperparameters, { 0: 0 })), Error);
    }
});
