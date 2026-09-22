import * as path from "path";
import Embedding from "./Embedding.js";
import GPTAttention from "./GPTAttention.js";
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
// Example: embed a sentence, then calculate its self-attention scores.
const exampleText = "Your journey starts with one step";
console.log(`index| Embedding example text: "${exampleText}"`);
const exampleEmbeddings = [[0.43, 0.15, 0.89],
    [0.55, 0.87, 0.66],
    [0.57, 0.85, 0.64],
    [0.22, 0.58, 0.33],
    [0.77, 0.25, 0.10],
    [0.05, 0.80, 0.55]];
console.log(`index| Got ${exampleEmbeddings.length} embedding vectors, each ${exampleEmbeddings[0]?.length ?? 0} values wide.`);
const attention = new GPTAttention();
attention.calculate(exampleEmbeddings);
