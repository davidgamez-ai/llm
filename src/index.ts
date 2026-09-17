import * as path from "path";
import BPETokenizer from "./BPETokenizer.js";

const filePath = path.join(import.meta.dirname, "..", "..", "data", "the-verdict.txt");

const tokenizer = new BPETokenizer();

const encoded:number[] = tokenizer.encode("glory <|endoftext|> I am a fish");
console.log(encoded);

const decoded:string = tokenizer.decode(encoded);
console.log(decoded);