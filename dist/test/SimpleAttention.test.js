import { test, beforeEach } from "node:test";
import assert from "node:assert/strict";
import SimpleAttention from "../SimpleAttention.js";
let attention;
beforeEach((t) => {
    // calculate() logs its progress via console.log; silence it so test
    // output only shows assertion results.
    t.mock.method(console, "log", () => { });
    attention = new SimpleAttention();
});
test("calculate() applies softmax to each token's row of dot products", () => {
    // A small, hand-worked embedding matrix (3 tokens, 2 values each) so the
    // raw dot products are easy to check by hand:
    //   row0: [1*1+0*0, 1*0+0*1, 1*1+0*1] = [1, 0, 1]
    //   row1: [0*1+1*0, 0*0+1*1, 0*1+1*1] = [0, 1, 1]
    //   row2: [1*1+1*0, 1*0+1*1, 1*1+1*1] = [1, 1, 2]
    const embeddings = [
        [1, 0],
        [0, 1],
        [1, 1],
    ];
    const rawScores = [
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 2],
    ];
    // Expected softmax of each row, computed independently of
    // SimpleAttention with the plain formula (no max-subtraction stability
    // trick), since the two are mathematically equivalent.
    const expected = rawScores.map((row) => {
        const exponentiated = row.map((score) => Math.exp(score));
        const total = exponentiated.reduce((sum, value) => sum + value, 0);
        return exponentiated.map((value) => value / total);
    });
    const weights = attention.calculate(embeddings);
    weights.forEach((row, i) => {
        row.forEach((value, j) => {
            assert.ok(Math.abs(value - expected[i][j]) < 1e-9);
        });
    });
});
test("calculate() returns a square matrix whose rows each sum to 1", () => {
    const embeddings = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [1, 0, 1],
    ];
    const weights = attention.calculate(embeddings);
    assert.equal(weights.length, embeddings.length);
    for (const row of weights) {
        assert.equal(row.length, embeddings.length);
        const total = row.reduce((sum, weight) => sum + weight, 0);
        assert.ok(Math.abs(total - 1) < 1e-9);
    }
});
test("calculate() of a single embedding vector normalizes to a weight of 1 on itself", () => {
    assert.deepEqual(attention.calculate([[3, 4]]), [[1]]);
});
test("calculate() of no embedding vectors returns an empty matrix", () => {
    assert.deepEqual(attention.calculate([]), []);
});
