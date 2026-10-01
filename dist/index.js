import * as path from "path";
import Hyperparameters from "./Hyperparameters.js";
import GPTModel from "./GPTModel.js";
const filePath = path.join(import.meta.dirname, "..", "data", "the-verdict.txt");
//Create hyperparameters with training set to false.
const hyperparameters = new Hyperparameters(768, 1024, 0.1, false, 12, 12, 2, false, true);
//New GPT model
const gptModel = new GPTModel(hyperparameters);
//Output number of parameters
console.log(`Number of parameters: ${gptModel.getParameterCount()}`);
//Test text
const text = "First test of GPT model";
//Tokenize text
