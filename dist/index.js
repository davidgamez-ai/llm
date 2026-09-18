import * as path from "path";
import Embedding from "./Embedding.js";
const filePath = path.join(import.meta.dirname, "..", "data", "the-verdict.txt");
// const tokenizer = new BPETokenizer();
// const encoded:number[] = tokenizer.encode("glory <|endoftext|> I am a fish");
// console.log(encoded);
// const decoded:string = tokenizer.decode(encoded);
// console.log(decoded);
// const trainer:Trainer = new Trainer();
// trainer.load(filePath);
const embeddings = new Embedding();
//Build the matrix
embeddings.build();
const tstEmbeddings = embeddings.getEmbedding("I am a fish");
