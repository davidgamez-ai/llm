"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const node_test_1 = require("node:test");
const strict_1 = __importDefault(require("node:assert/strict"));
const fs = __importStar(require("fs"));
const os = __importStar(require("os"));
const path = __importStar(require("path"));
const Tokenizer_1 = __importDefault(require("../src/Tokenizer"));
// A small, fixed corpus whose tokenization is worked out by hand below, so
// every test can assert on exact ids instead of relative/derived values.
//
// Splitting on whitespace and punctuation yields, in order:
//   Hello , world ! Hello again -- it ' s a test .
// The unique tokens, sorted alphabetically (JS default string sort, so
// punctuation and uppercase letters sort before lowercase letters), are:
//   0:"!" 1:"'" 2:"," 3:"--" 4:"." 5:"Hello" 6:"a" 7:"again" 8:"it" 9:"s"
//   10:"test" 11:"world"
const FIXTURE_TEXT = "Hello, world! Hello again -- it's a test.";
let fixtureDir;
let tokenizer;
function writeFixture(dir, contents) {
    const filePath = path.join(dir, "fixture.txt");
    fs.writeFileSync(filePath, contents, "utf-8");
    return filePath;
}
(0, node_test_1.beforeEach)((t) => {
    // train() logs its progress via console.log; silence it so test output
    // only shows assertion results.
    t.mock.method(console, "log", () => { });
    fixtureDir = fs.mkdtempSync(path.join(os.tmpdir(), "tokenizer-test-"));
    tokenizer = new Tokenizer_1.default();
    tokenizer.train(writeFixture(fixtureDir, FIXTURE_TEXT));
});
(0, node_test_1.afterEach)(() => {
    fs.rmSync(fixtureDir, { recursive: true, force: true });
});
(0, node_test_1.test)("train() assigns ids 0..n-1 to unique tokens in alphabetical order", () => {
    const ids = tokenizer.encode("! ' , -- . Hello a again it s test world");
    strict_1.default.deepEqual(ids, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]);
});
(0, node_test_1.test)("encode() splits punctuation into separate tokens from words", () => {
    strict_1.default.deepEqual(tokenizer.encode("Hello, world!"), [5, 2, 11, 0]);
});
(0, node_test_1.test)("encode() collapses repeated tokens to the same id", () => {
    const [firstHello] = tokenizer.encode("Hello");
    const [secondHello] = tokenizer.encode("Hello again");
    strict_1.default.equal(firstHello, secondHello);
});
(0, node_test_1.test)("encode() is case-sensitive", () => {
    strict_1.default.throws(() => tokenizer.encode("hello"), /Unknown token: "hello"/);
});
(0, node_test_1.test)("encode() throws on a token that was never seen during training", () => {
    strict_1.default.throws(() => tokenizer.encode("banana"), /Unknown token: "banana"/);
});
(0, node_test_1.test)("encode() throws naming the first unrecognized token in a longer string", () => {
    strict_1.default.throws(() => tokenizer.encode("Hello banana world"), /Unknown token: "banana"/);
});
(0, node_test_1.test)("decode() converts ids back to their tokens, space-separated", () => {
    strict_1.default.equal(tokenizer.decode([5, 2, 11, 0]), "Hello , world !");
});
(0, node_test_1.test)("decode() of an empty array returns an empty string", () => {
    strict_1.default.equal(tokenizer.decode([]), "");
});
(0, node_test_1.test)("encode() and decode() round-trip a sentence's tokens", () => {
    const sentence = "Hello again -- it's a test.";
    const ids = tokenizer.encode(sentence);
    strict_1.default.equal(tokenizer.decode(ids), "Hello again -- it ' s a test .");
});
(0, node_test_1.test)("train() replaces the previous vocabulary rather than merging it", () => {
    const secondFixtureDir = fs.mkdtempSync(path.join(os.tmpdir(), "tokenizer-test-"));
    try {
        tokenizer.train(writeFixture(secondFixtureDir, "cats and dogs"));
        strict_1.default.throws(() => tokenizer.encode("Hello"), /Unknown token: "Hello"/);
        strict_1.default.deepEqual(tokenizer.encode("cats and dogs"), [1, 0, 2]);
    }
    finally {
        fs.rmSync(secondFixtureDir, { recursive: true, force: true });
    }
});
