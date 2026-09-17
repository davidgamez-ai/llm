import { test, beforeEach, afterEach } from "node:test";
import assert from "node:assert/strict";
import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import Tokenizer from "../src/Tokenizer.js";
// A small, fixed corpus whose tokenization is worked out by hand below, so
// every test can assert on exact ids instead of relative/derived values.
//
// Splitting on whitespace and punctuation yields, in order:
//   Hello , world ! Hello again -- it ' s a test .
// The unique tokens, sorted alphabetically (JS default string sort, so
// punctuation and uppercase letters sort before lowercase letters), are:
//   0:"!" 1:"'" 2:"," 3:"--" 4:"." 5:"Hello" 6:"a" 7:"again" 8:"it" 9:"s"
//   10:"test" 11:"world"
// The special tokens are appended after that, so 12:"<|unk|>" and
// 13:"<|endoftext|>".
const FIXTURE_TEXT = "Hello, world! Hello again -- it's a test.";
const UNKNOWN_TOKEN_ID = 12;
let fixtureDir;
let tokenizer;
function writeFixture(dir, contents) {
    const filePath = path.join(dir, "fixture.txt");
    fs.writeFileSync(filePath, contents, "utf-8");
    return filePath;
}
beforeEach((t) => {
    // train() logs its progress via console.log; silence it so test output
    // only shows assertion results.
    t.mock.method(console, "log", () => { });
    fixtureDir = fs.mkdtempSync(path.join(os.tmpdir(), "tokenizer-test-"));
    tokenizer = new Tokenizer();
    tokenizer.train(writeFixture(fixtureDir, FIXTURE_TEXT));
});
afterEach(() => {
    fs.rmSync(fixtureDir, { recursive: true, force: true });
});
test("train() assigns ids 0..n-1 to unique tokens in alphabetical order", () => {
    const ids = tokenizer.encode("! ' , -- . Hello a again it s test world");
    assert.deepEqual(ids, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]);
});
test("encode() splits punctuation into separate tokens from words", () => {
    assert.deepEqual(tokenizer.encode("Hello, world!"), [5, 2, 11, 0]);
});
test("encode() collapses repeated tokens to the same id", () => {
    const [firstHello] = tokenizer.encode("Hello");
    const [secondHello] = tokenizer.encode("Hello again");
    assert.equal(firstHello, secondHello);
});
test("encode() is case-sensitive", () => {
    assert.deepEqual(tokenizer.encode("hello"), [UNKNOWN_TOKEN_ID]);
});
test("encode() substitutes the <|unk|> id for a token that was never seen during training", () => {
    assert.deepEqual(tokenizer.encode("banana"), [UNKNOWN_TOKEN_ID]);
});
test("encode() substitutes the <|unk|> id in place, alongside known tokens", () => {
    assert.deepEqual(tokenizer.encode("Hello banana world"), [
        5,
        UNKNOWN_TOKEN_ID,
        11,
    ]);
});
test("decode() converts ids back to their tokens, space-separated", () => {
    assert.equal(tokenizer.decode([5, 2, 11, 0]), "Hello , world !");
});
test("decode() of an empty array returns an empty string", () => {
    assert.equal(tokenizer.decode([]), "");
});
test("encode() and decode() round-trip a sentence's tokens", () => {
    const sentence = "Hello again -- it's a test.";
    const ids = tokenizer.encode(sentence);
    assert.equal(tokenizer.decode(ids), "Hello again -- it ' s a test .");
});
test("train() appends the special tokens after the regular vocabulary", () => {
    // The fixture has 12 unique regular tokens (ids 0-11), so the special
    // tokens should continue the sequence at 12 and 13.
    assert.deepEqual(tokenizer.encode("<|unk|> <|endoftext|>"), [12, 13]);
});
test("train() replaces the previous vocabulary rather than merging it", () => {
    const secondFixtureDir = fs.mkdtempSync(path.join(os.tmpdir(), "tokenizer-test-"));
    try {
        tokenizer.train(writeFixture(secondFixtureDir, "cats and dogs"));
        // New vocabulary: 0:"and" 1:"cats" 2:"dogs" 3:"<|unk|>" 4:"<|endoftext|>"
        assert.deepEqual(tokenizer.encode("Hello"), [3]);
        assert.deepEqual(tokenizer.encode("cats and dogs"), [1, 0, 2]);
    }
    finally {
        fs.rmSync(secondFixtureDir, { recursive: true, force: true });
    }
});
