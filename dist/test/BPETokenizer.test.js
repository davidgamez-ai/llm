import { test } from "node:test";
import assert from "node:assert/strict";
import BPETokenizer from "../BPETokenizer.js";
test("encode() returns an array of numeric token ids", () => {
    const tokenizer = new BPETokenizer();
    const ids = tokenizer.encode("Hello, world!");
    assert.ok(Array.isArray(ids));
    assert.ok(ids.length > 0);
    assert.ok(ids.every((id) => Number.isInteger(id)));
});
test("encode() and decode() round-trip arbitrary text, including words never seen in any training corpus", () => {
    const tokenizer = new BPETokenizer();
    const text = "supercalifragilisticexpialidocious, wasn't it?";
    assert.equal(tokenizer.decode(tokenizer.encode(text)), text);
});
test("decode() of an empty array returns an empty string", () => {
    const tokenizer = new BPETokenizer();
    assert.equal(tokenizer.decode([]), "");
});
test("endOfTextId is GPT-2's <|endoftext|> id, 50256", () => {
    assert.equal(new BPETokenizer().endOfTextId, 50256);
});
test("encode() turns <|endoftext|> into a single endOfTextId wherever it appears", () => {
    const tokenizer = new BPETokenizer();
    const eot = BPETokenizer.END_OF_TEXT;
    // At the start, in the middle (which gpt-tokenizer 4.0.0 gets wrong on its own), at the end, and repeated.
    for (const text of [`${eot}Hello`, `Hello${eot}World`, `Hello${eot}`, `A${eot}B${eot}${eot}C`]) {
        const ids = tokenizer.encode(text);
        const expectedCount = text.split(eot).length - 1;
        assert.equal(ids.filter((id) => id === tokenizer.endOfTextId).length, expectedCount, text);
        assert.equal(tokenizer.decode(ids), text, text);
    }
});
test("encode() of joined documents equals the documents encoded separately, joined by endOfTextId", () => {
    const tokenizer = new BPETokenizer();
    const first = "The first document.";
    const second = " The second document.";
    assert.deepEqual(tokenizer.encode(`${first}${BPETokenizer.END_OF_TEXT}${second}`), [...tokenizer.encode(first), tokenizer.endOfTextId, ...tokenizer.encode(second)]);
});
test("encode() of text without <|endoftext|> is unchanged", () => {
    assert.deepEqual(new BPETokenizer().encode("Hello World"), [15496, 2159]);
});
