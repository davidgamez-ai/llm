import * as path from "path";
import Hyperparameters from "./Hyperparameters.js";
import Generator from "./Generator.js";
import Trainer from "./Trainer.js";

const filePath = path.join(import.meta.dirname, "..", "data", "the-verdict.txt");

//Create hyperparameters with training set to false and a random seed.
const hyperparameters = new Hyperparameters();
hyperparameters.training = false;
hyperparameters.seed = 123;

//Trainer
const trainer:Trainer = new Trainer(hyperparameters);
trainer.load('data/the-verdict.txt');
trainer.train();


//Generator
// const generator: Generator = new Generator(hyperparameters);

// //Starting text
// const text:string = "First test of GPT model";

// //Call generator to build text
// const generatedText:string = generator.generate(text);

// //Output result
// console.log(`Final generated text: "${generatedText}".`);

