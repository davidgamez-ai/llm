import * as fs from "fs";

// Special tokens appended to the end of the vocabulary after training, in
// this order, so their ids always follow every regular token's id.
const UNKNOWN_TOKEN = "<|unk|>";
const END_OF_TEXT_TOKEN = "<|endoftext|>";
const SPECIAL_TOKENS = [UNKNOWN_TOKEN, END_OF_TEXT_TOKEN];

class Tokenizer {
  // Maps each unique token to the id assigned to it during training.
  private tokenToId: Map<string, number> = new Map();

  // Splits text on whitespace and punctuation, keeping the punctuation
  // characters as their own tokens (the capturing group makes split()
  // retain them), then drops empty tokens produced by whitespace runs and
  // leading/trailing splits. Used by both train() and encode() so they
  // tokenize text identically.
  private tokenize(text: string): string[] {
    return text
      .split(/([,.:;?_!"()']|--|\s)/)
      .map((token) => token.trim())
      .filter((token) => token.length > 0);
  }

  // Reads the given file, builds the vocabulary from it and stores a
  // token <-> id mapping on the instance. The id of each token is its
  // position in the alphabetically sorted list of unique tokens.
  train(filePath: string): void {
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
  encode(text: string): number[] {
    const tokens = this.tokenize(text);
    const unknownTokenId = this.tokenToId.get(UNKNOWN_TOKEN);

    return tokens.map((token) => this.tokenToId.get(token) ?? unknownTokenId!);
  }

  // Looks up the token for each id in the tokenToId mapping (each token's
  // id equals its position when the map's keys are listed in insertion
  // order) and joins the resulting words with whitespace.
  decode(ids: number[]): string {
    const idToToken = [...this.tokenToId.keys()];

    return ids.map((id) => idToToken[id]).join(" ");
  }
}

export default Tokenizer;
