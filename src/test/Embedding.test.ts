import { test } from "node:test";
import assert from "node:assert/strict";
import Hyperparameters from "../Hyperparameters.js";
import Embedding from "../Embedding.js";
import { checkGradients, randomMatrix } from "./GradientCheck.js";

/*
 * Builds a small Embedding (embeddingSize 4, contextLength 8, one attention
 * head) so the tests run quickly. The vocabulary size always comes from the
 * tokenizer.
 */
const buildEmbedding = (): Embedding => {
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

// Token ids for the backpropagation tests. Token 5 appears twice, at positions 0 and 2.
const TOKEN_IDS: number[] = [5, 9, 5, 2];

test("backpropagation gives the same token and position embedding gradients as the numerical gradients", () => {
  const embedding = buildEmbedding();
  const tokens = embedding.textEmbeddingMatrix;

  /*
   * Only the rows of tokens in the window can have a gradient, so the check
   * is given just those rows: getRow() returns views onto the matrices, so
   * nudging a value in a row nudges it in the matrix.
   */
  const tokenRow = (id: number) => ({
    name: `token ${id} row`,
    values: tokens.getRow(id),
    gradients: () => embedding.textEmbeddingGradients.getRow(id),
  });

  checkGradients({
    forward: () => embedding.getEmbeddingFromTokens(TOKEN_IDS),
    backward: (outputErrors) => {
      embedding.backpropagate(outputErrors);
      embedding.calculateGradients(false);
    },
    parameters: [
      tokenRow(5),
      tokenRow(9),
      tokenRow(2),
      { name: "positionEmbeddingMatrix", values: embedding.positionEmbeddingMatrix.data, gradients: () => embedding.positionEmbeddingGradients.data },
    ],
  });
});

test("a token that appears twice gets the sum of both positions' errors; tokens not in the window get none", () => {
  const embedding = buildEmbedding();
  embedding.getEmbeddingFromTokens(TOKEN_IDS);
  const errors = randomMatrix(4, 4, 7);

  embedding.backpropagate(errors);
  embedding.calculateGradients(false);

  // Gradients are stored as 32-bit floats, rounded after each addition, so they match to about 7 digits.
  const gradients = embedding.textEmbeddingGradients;
  for (let i = 0; i < 4; i++) {
    assert.ok(Math.abs(gradients.get(5, i) - (errors[0][i] + errors[2][i])) < 1e-6);
    assert.ok(Math.abs(gradients.get(9, i) - errors[1][i]) < 1e-6);
  }

  // Every row other than tokens 5, 9 and 2 is exactly zero.
  for (let row = 0; row < gradients.rows; row++) {
    if (![5, 9, 2].includes(row)) {
      assert.ok(gradients.getRow(row).every((value) => value === 0), `row ${row} has a gradient`);
    }
  }
});

test("only the positions used have position gradients", () => {
  const embedding = buildEmbedding();
  // contextLength is 8, but only the first 4 positions are used.
  embedding.getEmbeddingFromTokens(TOKEN_IDS);
  embedding.backpropagate(randomMatrix(4, 4, 8));
  embedding.calculateGradients(false);

  for (let position = 4; position < 8; position++) {
    assert.ok(embedding.positionEmbeddingGradients.getRow(position).every((value) => value === 0));
  }
});

test("backpropagate() rejects errors for a different number of tokens", () => {
  const embedding = buildEmbedding();
  embedding.getEmbeddingFromTokens(TOKEN_IDS);

  assert.throws(() => embedding.backpropagate(randomMatrix(3, 4, 1)), RangeError);
});
