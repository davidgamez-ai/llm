import * as path from "path";
import Tokenizer from "./Tokenizer";

const filePath = path.join(__dirname, "..", "..", "data", "the-verdict.txt");

const tokenizer = new Tokenizer();
tokenizer.train(filePath);

const encoded:number[] = tokenizer.encode("glory <|endoftext|> I am a fish");
console.log(encoded);

const decoded:string = tokenizer.decode(encoded);
console.log(decoded);