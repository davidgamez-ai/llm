import * as path from "path";
import BPETokenizer from "./BPETokenizer.js";
import Trainer from "./Trainer.js";
import Embedding from "./Embedding.js";
import SimpleAttention from "./SimpleAttention.js";
import Context from "./Context.js";
import GPTAttention from "./GPTAttention.js";
import MultiHeadAttention from "./MultiHeadAttention.js";
import Hyperparameters from "./Hyperparameters.js";
import GPTModel from "./GPTModel.js";
import Decoder from "./Decoder.js";

const filePath = path.join(import.meta.dirname, "..", "data", "the-verdict.txt");

//Create hyperparameters with training set to false and a random seed.
const hyperparameters = new Hyperparameters();
hyperparameters.training = false;
hyperparameters.seed = 123;

//Tokenizer
const bpeTokenizer:BPETokenizer = new BPETokenizer();

//New GPT model
const gptModel:GPTModel = new GPTModel(hyperparameters);

//Decoder to interpret output in generation mode
const decoder:Decoder = new Decoder(hyperparameters);

//Starting text
let text:string = "First test of GPT model";

//Generate 2 tokens
for(let t:number=0; t<2; ++t){
    //Convert text to tokens
    const tokens:number[] = bpeTokenizer.encode(text);

    //Feed text into model
    const tokenProbs:number[][] = gptModel.calculate(tokens);

    //Check dimensions of returned array
    if(tokenProbs.length !== 1)
        throw "First dimension of tokenProbs should be 1";

    //Decode output
    const nextText = decoder.greedy(tokenProbs[0]);

    //Add output to text
    text += nextText;

    //Logging
    console.log(`Text: ${text}. Counter: ${t}.`);
}

//Output result
console.log(`Final generated text: "${text}".`);



