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

const filePath = path.join(import.meta.dirname, "..", "data", "the-verdict.txt");

//Create hyperparameters with training set to false.
const hyperparameters = new Hyperparameters(
    768,
    1024,
    0.1,
    false,
    12,
    12,
     2,
    false,
    true
);

//New GPT model
const gptModel = new GPTModel(hyperparameters);

//Output number of parameters
console.log(`Number of parameters: ${gptModel.getParameterCount()}`)

//Test text
const text = "First test of GPT model";

//Tokenize text



