import * as fs from "fs";
import BPETokenizer from "./BPETokenizer.js";

class Trainer {
  // Reads the given file, encodes its contents using the BPE tokenizer,
  // and logs the number of resulting tokens.
  load(filePath: string): void {
    const text = fs.readFileSync(filePath, "utf-8");
    const tokenizer = new BPETokenizer();
    const encoded = tokenizer.encode(text);

    console.log(`Encoded text length: ${encoded.length}`);
  }
}

export default Trainer;
