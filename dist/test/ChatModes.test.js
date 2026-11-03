import { test } from "node:test";
import assert from "node:assert/strict";
import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import { PassThrough, Writable } from "stream";
import AlpacaDataset from "../AlpacaDataset.js";
import BPETokenizer from "../BPETokenizer.js";
import GPTConsole from "../GPTConsole.js";
import GPTModel from "../GPTModel.js";
import Hyperparameters from "../Hyperparameters.js";
import ModelManager from "../ModelManager.js";
import TinyStoriesDataset from "../TinyStoriesDataset.js";
import Trainer from "../Trainer.js";
const INSTRUCTION = "Name a colour.";
const ANSWER = "Blue is a colour.";
const STORY_START = "Once upon a time";
const STORY_END = " there was a cat.";
/** The number of tokens the story model generates in Standard chat, which is more than are left in the story. */
const OUTPUT_LENGTH = 10;
/**
 * Trains a model to remember a dataset and saves it, so that its replies are
 * predictable. The real console can then load it and show what each chat mode
 * does with a model that has learned to produce the end token, as a properly
 * trained one should.
 *
 * @param dataset - The dataset to remember.
 * @param trainingWindowLength - The length of the training windows, or undefined for one window per example.
 * @param numberTrainingExamples - The number of examples to train on.
 * @param numberValidationExamples - The number of examples to validate on.
 * @returns The path of the saved model.
 */
const trainedModelFile = (dataset, trainingWindowLength, numberTrainingExamples, numberValidationExamples) => {
    const hyperparameters = new Hyperparameters(16, 128, 0, Hyperparameters.TRAINING_MODE, 2, 1, 1);
    hyperparameters.seed = 42;
    hyperparameters.learningRate = 0.01;
    hyperparameters.numberEpochs = 60;
    hyperparameters.outputLength = OUTPUT_LENGTH;
    hyperparameters.numberTrainingExamples = numberTrainingExamples;
    hyperparameters.numberValidationExamples = numberValidationExamples;
    if (trainingWindowLength !== undefined) {
        hyperparameters.trainingWindowLength = trainingWindowLength;
    }
    const model = new GPTModel(hyperparameters);
    const trainer = new Trainer(hyperparameters, model);
    trainer.load(dataset);
    trainer.train();
    const modelFile = path.join(fs.mkdtempSync(path.join(os.tmpdir(), "chat-modes-test-")), "remembers.safetensors");
    new ModelManager().save(model, modelFile);
    return modelFile;
};
/**
 * Trains a model to remember one Alpaca example, including the template and
 * the end token.
 *
 * @returns The path of the saved model.
 */
const modelRememberingAlpaca = () => {
    const dataFile = path.join(fs.mkdtempSync(path.join(os.tmpdir(), "chat-modes-test-")), "alpaca.json");
    // Two copies, since one example is held out for validation.
    const example = { instruction: INSTRUCTION, input: "", output: ANSWER };
    fs.writeFileSync(dataFile, JSON.stringify([example, example]), "utf-8");
    return trainedModelFile(new AlpacaDataset(dataFile), undefined, 1, 1);
};
/**
 * Trains a model to remember a story and the end token that follows it. Each
 * window holds exactly one story and its end token, so the model learns the
 * story from the first position, which is where text typed in chat starts.
 *
 * @returns The path of the saved model.
 */
const modelRememberingStory = () => {
    const story = STORY_START + STORY_END;
    const folder = fs.mkdtempSync(path.join(os.tmpdir(), "chat-modes-test-"));
    const writeCsv = (name, count) => {
        const filePath = path.join(folder, name);
        fs.writeFileSync(filePath, "text\n" + `"${story}"\n`.repeat(count), "utf-8");
        return filePath;
    };
    const windowLength = new BPETokenizer().encode(story).length + 1;
    // Two validation stories, since one is shorter than a window and the one token that follows it.
    return trainedModelFile(new TinyStoriesDataset(writeCsv("training.csv", 4), writeCsv("validation.csv", 2)), windowLength, 4, 2);
};
/**
 * Loads a saved model into a real console, chats in one mode and returns the
 * model's reply. The console's greedy sampling makes the reply repeatable.
 *
 * @param t - The test context, used to silence console.log.
 * @param modelFile - The path of the saved model.
 * @param mode - The number of the chat mode, as typed.
 * @param typed - The line typed in chat.
 * @returns The text of the reply.
 */
const replyInMode = async (t, modelFile, mode, typed) => {
    t.mock.method(console, "log", () => { });
    const input = new PassThrough();
    input.end(["1", "4", modelFile, "7", "greedy", "8", mode, typed, "quit", "quit"].map((line) => line + "\n").join(""));
    let written = "";
    const output = new Writable({
        write(chunk, _encoding, callback) {
            written += chunk.toString();
            callback();
        },
    });
    // The model created at startup is replaced by the loaded one, so its size does not matter.
    const hyperparametersForSize = () => new Hyperparameters(8, 4, 0, Hyperparameters.TRAINING_MODE, 2, 1, 1);
    await new GPTConsole({ hyperparametersForSize, input, output }).run();
    const reply = written.match(/Model: ([\s\S]*?)\nYou: /);
    assert.ok(reply !== null, written);
    return reply[1];
};
test("Alpaca chat puts the instruction in the template, gives only the answer and stops at the end token", async (t) => {
    assert.equal(await replyInMode(t, modelRememberingAlpaca(), "3", INSTRUCTION), ANSWER);
});
test("Standard and Early stop chat continue the same text differently", async (t) => {
    const modelFile = modelRememberingStory();
    await t.test("Early stop gives the rest of the story and stops at the end token", async (t) => {
        assert.equal(await replyInMode(t, modelFile, "2", STORY_START), STORY_END);
    });
    await t.test("Standard gives the rest of the story, shows the end token and runs on past it", async (t) => {
        const reply = await replyInMode(t, modelFile, "1", STORY_START);
        const expectedStart = `${STORY_END}${BPETokenizer.END_OF_TEXT}`;
        assert.ok(reply.startsWith(expectedStart), reply);
        assert.ok(reply.length > expectedStart.length, reply);
    });
});
test("a prompt tokenises the same alone as at the start of a whole example, so a model sees in chat what it saw in training", () => {
    const tokenizer = new BPETokenizer();
    const prompt = tokenizer.encode(AlpacaDataset.formatPrompt(INSTRUCTION));
    for (const output of [ANSWER, ` ${ANSWER}`, "1. Blue\n2. Red"]) {
        const example = tokenizer.encode(AlpacaDataset.formatPrompt(INSTRUCTION) + output);
        assert.deepEqual(example.slice(0, prompt.length), prompt, JSON.stringify(output));
    }
});
