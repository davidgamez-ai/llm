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
// //Create hyperparameters in generation mode with a random seed. Trainer.train() switches to training mode while it runs.
const hyperparameters = new Hyperparameters();
hyperparameters.mode = Hyperparameters.GENERATION_MODE;
hyperparameters.seed = 123;
//Build a single model, shared by the trainer and the generator so that generation uses the trained weights.
const model = new GPTModel(hyperparameters);
//Build generator
const generator = new Generator(hyperparameters, model);
//Test text
const text = "It was not till three years later";
//Output completion before model is trained.
//Call generator to build text
let generatedText = generator.generate(text);
console.log(`Generated text before training: "${generatedText}".`);
// Train model
const trainer = new Trainer(hyperparameters, model);
trainer.load(filePath);
trainer.train();
//Output completion after model is trained.
//Call generator to build text
generatedText = generator.generate(text);
console.log(`Generated text after training: "${generatedText}".`);
