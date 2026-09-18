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
