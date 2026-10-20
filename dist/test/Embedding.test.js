import { test } from "node:test";
import assert from "node:assert/strict";
import Hyperparameters from "../Hyperparameters.js";
import Embedding from "../Embedding.js";
/*
 * Builds a small Embedding (embeddingSize 4, contextLength 8, one attention
 * head) so the tests run quickly. The vocabulary size always comes from the
 * tokenizer.
 */
const buildEmbedding = () => {
    const hyperparameters = new Hyperparameters(4, 8, 0.1, Hyperparameters.TRAINING_MODE, 1);
    hyperparameters.seed = 1;
    const embedding = new Embedding(hyperparameters);
    embedding.build();
    return embedding;
};
test("getEmbeddingFromTokens() accepts the first and last ids in the vocabulary", () => {
    const embedding = buildEmbedding();
    const result = embedding.getEmbeddingFromTokens([0, embedding.vocabSize - 1]);
    assert.equal(result.length, 2);
    assert.ok(result.every((vector) => vector.length === 4 && vector.every(Number.isFinite)));
});
test("getEmbeddingFromTokens() rejects ids outside the vocabulary, fractions and NaN with a RangeError", () => {
    const embedding = buildEmbedding();
    for (const id of [-1, embedding.vocabSize, 1.5, NaN, Infinity]) {
        assert.throws(() => embedding.getEmbeddingFromTokens([0, id]), RangeError, `accepted ${id}`);
    }
});
test("getEmbeddingFromTokens() error names the bad id and its position", () => {
    const embedding = buildEmbedding();
    assert.throws(() => embedding.getEmbeddingFromTokens([0, 1, 99999]), /Token id 99999 at position 2/);
});
