import * as path from "path";
import BPETokenizer from "./BPETokenizer.js";
const filePath = path.join(import.meta.dirname, "..", "..", "data", "the-verdict.txt");
const tokenizer = new BPETokenizer();
const encoded = tokenizer.encode("Akwirw ier");
console.log(encoded);
const decoded = tokenizer.decode(encoded);
console.log(decoded);
