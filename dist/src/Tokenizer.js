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
Object.defineProperty(exports, "__esModule", { value: true });
const fs = __importStar(require("fs"));
// Special tokens appended to the end of the vocabulary after training, in
// this order, so their ids always follow every regular token's id.
const UNKNOWN_TOKEN = "<|unk|>";
const END_OF_TEXT_TOKEN = "<|endoftext|>";
const SPECIAL_TOKENS = [UNKNOWN_TOKEN, END_OF_TEXT_TOKEN];
class Tokenizer {
    // Maps each unique token to the id assigned to it during training.
    tokenToId = new Map();
    // Splits text on whitespace and punctuation, keeping the punctuation
    // characters as their own tokens (the capturing group makes split()
    // retain them), then drops empty tokens produced by whitespace runs and
    // leading/trailing splits. Used by both train() and encode() so they
    // tokenize text identically.
    tokenize(text) {
        return text
            .split(/([,.:;?_!"()']|--|\s)/)
            .map((token) => token.trim())
            .filter((token) => token.length > 0);
    }
    // Reads the given file, builds the vocabulary from it and stores a
    // token <-> id mapping on the instance. The id of each token is its
    // position in the alphabetically sorted list of unique tokens.
    train(filePath) {
        const text = fs.readFileSync(filePath, "utf-8");
        const tokens = this.tokenize(text);
        console.log(tokens.map((token) => `"${token}"`).join(", "));
        // Deduplicate tokens via a Set (insertion order doesn't matter here, only
        // uniqueness) to determine the size of the vocabulary, then sort
        // alphabetically so the vocabulary has a stable, readable ordering.
        const uniqueTokens = [...new Set(tokens)].sort();
        console.log(uniqueTokens.map((token) => `"${token}"`).join(", "));
        console.log(`Number of unique tokens: ${uniqueTokens.length}`);
        this.tokenToId = new Map(uniqueTokens.map((token, id) => [token, id]));
        // Append the special tokens after the regular vocabulary, continuing
        // the id sequence rather than restarting it.
        for (const specialToken of SPECIAL_TOKENS) {
            this.tokenToId.set(specialToken, this.tokenToId.size);
        }
        const firstTenEntries = [...this.tokenToId.entries()]
            .slice(0, 10)
            .map(([token, id]) => `('${token}', ${id})`)
            .join(", ");
        console.log(firstTenEntries);
    }
    // Tokenizes the given text the same way train() does and looks up each
    // token's id in the vocabulary, substituting the <|unk|> token's id for
    // any token that was never seen during training.
    encode(text) {
        const tokens = this.tokenize(text);
        const unknownTokenId = this.tokenToId.get(UNKNOWN_TOKEN);
        return tokens.map((token) => this.tokenToId.get(token) ?? unknownTokenId);
    }
    // Looks up the token for each id in the tokenToId mapping (each token's
    // id equals its position when the map's keys are listed in insertion
    // order) and joins the resulting words with whitespace.
    decode(ids) {
        const idToToken = [...this.tokenToId.keys()];
        return ids.map((id) => idToToken[id]).join(" ");
    }
}
exports.default = Tokenizer;
