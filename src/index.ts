import * as path from "path";
import Hyperparameters from "./Hyperparameters.js";
import GPTModel from "./GPTModel.js";
import Generator from "./Generator.js";
import Trainer from "./Trainer.js";

/*
 * Path to the training text, resolved from the location of this compiled file
 * (dist/index.js) rather than the current working directory, so the program
 * finds the data however it is launched.
 */
const filePath = path.join(import.meta.dirname, "..", "data", "the-verdict.txt");

//Create hyperparameters in generation mode with a random seed. Trainer.train() switches to training mode while it runs.
const hyperparameters = new Hyperparameters();
hyperparameters.mode = Hyperparameters.GENERATION_MODE;
hyperparameters.seed = 123;

//Build a single model, shared by the trainer and the generator so that generation uses the trained weights.
const model:GPTModel = new GPTModel(hyperparameters);

//Trainer
const trainer:Trainer = new Trainer(hyperparameters, model);
trainer.load(filePath);
trainer.train();


//Generator
// const generator: Generator = new Generator(hyperparameters, model);

// //Starting text
// const text:string = "First test of GPT model";

// //Call generator to build text
// const generatedText:string = generator.generate(text);

// //Output result
// console.log(`Final generated text: "${generatedText}".`);

