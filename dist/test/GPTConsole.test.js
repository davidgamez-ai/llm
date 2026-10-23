import { test } from "node:test";
import assert from "node:assert/strict";
import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import { PassThrough, Writable } from "stream";
import Hyperparameters from "../Hyperparameters.js";
import GPTConsole from "../GPTConsole.js";
import ModelManager from "../ModelManager.js";
/** Writes a CSV file in the TinyStories format (a "text" column of quoted stories) and returns its path. */
const writeStoriesCsv = (stories) => {
    const folder = fs.mkdtempSync(path.join(os.tmpdir(), "console-test-"));
    const filePath = path.join(folder, "stories.csv");
    fs.writeFileSync(filePath, "text\n" + stories.map((story) => `"${story}"`).join("\n") + "\n", "utf-8");
    return filePath;
};
// Enough for three runs of 8 stories each, since each run continues with new stories.
const STORIES = Array.from({ length: 24 }, (_, i) => `Once upon a time there was a cat called Tom ${i}. The end.`);
/** The hours each test's console was asked to build hyperparameters for, in order. */
let requestedHours = [];
/**
 * A tiny stand-in for Hyperparameters.forReducedTraining, so the tests run
 * quickly. Like the real one, it uses a wider model from 20 hours: 8 values
 * wide below 20 hours and 16 from 20 hours on. It records the hours asked for.
 */
const tinyHyperparametersForHours = (hours) => {
    requestedHours.push(hours);
    const hyperparameters = new Hyperparameters(hours < 20 ? 8 : 16, 4, 0.1, Hyperparameters.TRAINING_MODE, 2, 1, 3);
    hyperparameters.seed = 42;
    hyperparameters.outputLength = 5;
    hyperparameters.numberTrainingStories = 8;
    hyperparameters.numberValidationStories = 2;
    return hyperparameters;
};
/**
 * Runs a GPTConsole with the tiny hyperparameters on the given lines of
 * typed input, and returns everything it wrote. The trainer's own progress
 * messages go to console.log, which is silenced.
 *
 * @param t - The test context, used to silence console.log.
 * @param typed - The lines the user types, in order.
 * @param trainingFilePath - The training CSV. Defaults to a small file of STORIES.
 * @param gpt2FilePath - The GPT-2 small weights. Defaults to the console's own default, models/gpt2-small.safetensors.
 */
const runConsole = async (t, typed, trainingFilePath = writeStoriesCsv(STORIES), gpt2FilePath) => {
    t.mock.method(console, "log", () => { });
    requestedHours = [];
    // Typed input: every line at once, then the end of the input.
    const input = new PassThrough();
    input.end(typed.map((line) => line + "\n").join(""));
    // Collect everything written.
    let written = "";
    const output = new Writable({
        write(chunk, _encoding, callback) {
            written += chunk.toString();
            callback();
        },
    });
    const gptConsole = new GPTConsole({
        hyperparametersForHours: tinyHyperparametersForHours,
        trainingFilePath,
        validationFilePath: writeStoriesCsv(STORIES.slice(0, 2)),
        gpt2FilePath,
        input,
        output,
    });
    await gptConsole.run();
    return written;
};
/** Counts how many times a pattern appears in the text. */
const count = (text, pattern) => text.match(new RegExp(pattern.source, "g"))?.length ?? 0;
test("the menu offers Train, Save model, Load model, Load GPT-2 small weights, Temperature, Sampling method and Chat, in that order, and quit ends the program", async (t) => {
    const written = await runConsole(t, ["quit"]);
    assert.match(written, /1\. Train\n\s+2\. Save model\n\s+3\. Load model\n\s+4\. Load GPT-2 small weights\n\s+5\. Temperature\n\s+6\. Sampling method\n\s+7\. Chat/);
    assert.match(written, /Goodbye\./);
});
test("chat shows the model's continuation of each sentence, and quit returns to the menu", async (t) => {
    const written = await runConsole(t, ["7", "Once upon a time", "", "The cat", "quit", "quit"]);
    // Two replies: one per sentence, none for the empty line.
    assert.equal(count(written, /Model: /), 2);
    assert.match(written, /has not been trained yet/);
    // The menu is shown again after leaving chat (twice in all).
    assert.equal(count(written, /7\. Chat/), 2);
    assert.match(written, /Goodbye\./);
});
test("train asks for the hours, passes them on, describes the plan and trains", async (t) => {
    const written = await runConsole(t, ["1", "2.5", "7", "Once upon a time", "quit", "quit"]);
    // The constructor builds the first model for 2.5 hours; the answer typed is passed on too.
    assert.deepEqual(requestedHours, [2.5, 2.5]);
    assert.match(written, /How many hours of training\?/);
    assert.match(written, /Training for about 2\.5 hours: 8 stories \(numbers 1 to 8\), 1 epoch, a model 8 values wide/);
    assert.match(written, /Expected to finish at about/);
    assert.match(written, /Training finished\. Training loss: \d+\.\d{3}\. Validation loss: \d+\.\d{3}\./);
    assert.match(written, /trained 1 time\b/);
    assert.doesNotMatch(written, /has not been trained yet/);
    assert.match(written, /Model: /);
});
test("the hours are asked for again until they are a number greater than 0 and no more than 100", async (t) => {
    const written = await runConsole(t, ["1", "abc", "2 hours", "0", "-3", "101", "Infinity", "1", "quit"]);
    assert.match(written, /"abc" is not a number of hours\./);
    assert.match(written, /"2 hours" is not a number of hours\./);
    assert.equal(count(written, /must be more than 0/), 2);
    assert.match(written, /The most the console accepts is 100 hours\./);
    assert.match(written, /"Infinity" is not a number of hours\./);
    // Asked seven times; trained once, with the first sensible answer.
    assert.equal(count(written, /How many hours of training\?/), 7);
    assert.deepEqual(requestedHours, [2.5, 1]);
    assert.match(written, /trained 1 time\b/);
});
test("pressing Enter or typing quit when asked for the hours cancels training", async (t) => {
    const written = await runConsole(t, ["1", "", "1", "QUIT", "quit"]);
    assert.equal(count(written, /Training cancelled\./), 2);
    assert.doesNotMatch(written, /Training finished/);
    assert.match(written, /trained 0 times/);
    assert.match(written, /Goodbye\./);
});
test("training again with a model of the same size continues training the same model, on the next stories", async (t) => {
    const written = await runConsole(t, ["1", "1", "1", "3", "1", "2", "quit"]);
    assert.equal(count(written, /Training finished/), 3);
    assert.match(written, /trained 3 times/);
    assert.doesNotMatch(written, /Start a new model\?/);
    // Each run starts where the last one finished.
    assert.match(written, /numbers 1 to 8\)[\s\S]*numbers 9 to 16\)[\s\S]*numbers 17 to 24\)/);
});
test("training for hours that need a different size of model asks first, and 'n' keeps the trained model", async (t) => {
    // 1 hour builds the narrow model; 20 hours needs the wider one.
    const written = await runConsole(t, ["1", "1", "1", "20", "n", "quit"]);
    assert.match(written, /needs a model 16 values wide, but the current model is 8 wide/);
    assert.match(written, /Training cancelled; the current model is unchanged\./);
    assert.equal(count(written, /Training finished/), 1);
    // The menu's last count is still 1: the trained model was kept.
    assert.match(written, /trained 1 time\b[\s\S]*trained 1 time\b[\s\S]*Goodbye/);
});
test("answering 'y' builds the new model, which starts its training count again", async (t) => {
    const written = await runConsole(t, ["1", "1", "1", "20", "y", "quit"]);
    // The new model's trainer starts again from the first story.
    assert.match(written, /Training for about 20 hours: 8 stories \(numbers 1 to 8\), 1 epoch, a model 16 values wide/);
    assert.equal(count(written, /Training finished/), 2);
    // A new model, trained once.
    assert.match(written, /trained 1 time\b[\s\S]*Goodbye/);
    assert.doesNotMatch(written, /trained 2 times/);
});
test("an untrained model is rebuilt at the new size without asking", async (t) => {
    const written = await runConsole(t, ["1", "20", "quit"]);
    assert.doesNotMatch(written, /Start a new model\?/);
    assert.match(written, /a model 16 values wide/);
    assert.match(written, /Training finished/);
});
test("quitting with an untrained model does not ask for confirmation", async (t) => {
    const written = await runConsole(t, ["7", "Hello", "quit", "quit"]);
    assert.doesNotMatch(written, /Quit anyway\?/);
    assert.match(written, /Goodbye\./);
});
test("quitting with a trained model asks for confirmation; 'n' stays in the program and 'y' quits", async (t) => {
    const written = await runConsole(t, ["1", "1", "quit", "n", "quit", "Y"]);
    assert.equal(count(written, /Quit anyway\? \(y\/n\)/), 2);
    assert.match(written, /The model has been trained 1 time, and its training will be lost when the program ends\./);
    // After "n" the menu is shown again: once at the start, once after training, once after declining.
    assert.equal(count(written, /7\. Chat/), 3);
    assert.equal(count(written, /Goodbye\./), 1);
    assert.match(written, /Quit anyway\? \(y\/n\): Goodbye\.\s*$/);
});
test("the program ends if the input ends while asking whether to quit", async (t) => {
    const written = await runConsole(t, ["1", "1", "quit"]);
    assert.match(written, /Quit anyway\?[\s\S]*Goodbye\./);
});
test("a training error is shown and the console returns to the menu", async (t) => {
    const written = await runConsole(t, ["1", "1", "quit"], path.join(os.tmpdir(), "no-such-folder", "missing.csv"));
    assert.match(written, /Training failed: .*no such file/);
    assert.match(written, /trained 0 times/);
    assert.match(written, /Goodbye\./);
});
test("quit is accepted in any case, and an unknown choice is reported", async (t) => {
    const written = await runConsole(t, ["8", "2", "", "QUIT", "Quit"]);
    assert.match(written, /"8" is not an option\./);
    assert.match(written, /Goodbye\./);
});
/*
 * Saving and loading models (menu options 2 and 3).
 */
/** Creates an empty temporary folder for model files and returns its path. */
const modelFolder = () => fs.mkdtempSync(path.join(os.tmpdir(), "console-model-test-"));
test("save asks for a file name, saves the model with its progress, and quitting then does not ask for confirmation", async (t) => {
    const filePath = path.join(modelFolder(), "trained.safetensors");
    const written = await runConsole(t, ["1", "1", "2", filePath, "quit"]);
    assert.match(written, /File to save the model to\?/);
    assert.match(written, /Saving the model \(\d+\.\d million parameters, \d+\.\d MB\) to /);
    assert.match(written, /Saved: 10% 20% 30% 40% 50% 60% 70% 80% 90% 100%\n/);
    assert.match(written, /Model saved in \d+\.\d seconds\./);
    assert.ok(fs.statSync(filePath).size > 0);
    assert.doesNotMatch(written, /Quit anyway\?/);
});
test("save adds .safetensors to a file name without an extension, and removes surrounding quotes", async (t) => {
    const folder = modelFolder();
    await runConsole(t, ["2", `"${path.join(folder, "model")}"`, "quit"]);
    assert.ok(fs.existsSync(path.join(folder, "model.safetensors")));
});
test("save asks before overwriting a file; 'n' asks for another name and leaves the file unchanged, 'y' overwrites", async (t) => {
    const folder = modelFolder();
    const existing = path.join(folder, "existing.safetensors");
    fs.writeFileSync(existing, "keep me");
    const other = path.join(folder, "other.safetensors");
    const declined = await runConsole(t, ["2", existing, "n", other, "quit"]);
    assert.match(declined, /A file already exists at .*existing\.safetensors\. Overwrite it\? \(y\/n\)/);
    assert.match(declined, /The existing file has not been changed\./);
    assert.equal(fs.readFileSync(existing, "utf-8"), "keep me");
    assert.ok(fs.existsSync(other));
    const accepted = await runConsole(t, ["2", existing, "y", "quit"]);
    assert.match(accepted, /Model saved/);
    assert.notEqual(fs.readFileSync(existing, "utf-8"), "keep me");
});
test("save asks again for a file name in a folder that does not exist, a folder name, or a name with invalid characters", async (t) => {
    const folder = modelFolder();
    const written = await runConsole(t, [
        "2",
        path.join(folder, "missing", "model.safetensors"),
        folder,
        path.join(folder, "bad?name.safetensors"),
        path.join(folder, "good.safetensors"),
        "quit",
    ]);
    assert.match(written, /The folder .*missing does not exist\. Please enter a different file name\./);
    assert.match(written, /is a folder, not a file\. Please enter a different file name\./);
    assert.match(written, /"bad\?name\.safetensors" contains characters that are not allowed in file names\./);
    assert.equal(count(written, /File to save the model to\?/), 4);
    assert.ok(fs.existsSync(path.join(folder, "good.safetensors")));
});
test("pressing Enter or typing quit when asked for a file name cancels saving or loading", async (t) => {
    const written = await runConsole(t, ["2", "", "3", "quit", "quit"]);
    assert.match(written, /Save cancelled\./);
    assert.match(written, /Load cancelled\./);
});
test("load replaces the model with a saved one, showing its progress, and chat uses it", async (t) => {
    const filePath = path.join(modelFolder(), "trained.safetensors");
    // Train and save, then start a new console and load the saved model.
    await runConsole(t, ["1", "1", "2", filePath, "quit"]);
    const written = await runConsole(t, ["3", filePath, "7", "Once upon a time", "quit", "quit"]);
    assert.match(written, /File to load the model from\?/);
    assert.match(written, /Loading the model from .*trained\.safetensors \(\d+\.\d MB\)/);
    assert.match(written, /Loaded: 10% 20% 30% 40% 50% 60% 70% 80% 90% 100%\n/);
    assert.match(written, /Model loaded in \d+\.\d seconds: \d+\.\d million parameters, a model 8 values wide with 1 transformer blocks\./);
    assert.match(written, /trained 0 times since it was loaded from trained\.safetensors/);
    assert.doesNotMatch(written, /has not been trained yet/);
    assert.match(written, /Model: /);
});
test("load asks again for a file name where there is no file, or a folder", async (t) => {
    const folder = modelFolder();
    const filePath = path.join(folder, "model.safetensors");
    await runConsole(t, ["2", filePath, "quit"]);
    const written = await runConsole(t, ["3", path.join(folder, "missing.safetensors"), folder, filePath, "quit"]);
    assert.match(written, /There is no file at .*missing\.safetensors\. Please enter a different file name\./);
    assert.match(written, /is a folder, not a file\. Please enter a different file name\./);
    assert.match(written, /Model loaded/);
});
test("load asks before losing unsaved training; 'n' keeps the current model", async (t) => {
    const filePath = path.join(modelFolder(), "model.safetensors");
    await runConsole(t, ["2", filePath, "quit"]);
    const written = await runConsole(t, ["1", "1", "3", filePath, "n", "quit", "y"]);
    assert.match(written, /trained 1 time since it was last saved, and its training will be lost if another model is loaded\./);
    assert.match(written, /Load cancelled; the current model is unchanged\./);
    assert.doesNotMatch(written, /Model loaded/);
    // Still unsaved, so quitting asks.
    assert.match(written, /Quit anyway\?/);
});
test("a file that is not a saved model is reported, and the current model is kept", async (t) => {
    const filePath = path.join(modelFolder(), "notes.txt");
    fs.writeFileSync(filePath, "not a model");
    const written = await runConsole(t, ["3", filePath, "quit"]);
    assert.match(written, /Loaded:\n/);
    assert.match(written, /Loading failed: .*not a valid safetensors file/);
    assert.match(written, /The current model is unchanged\./);
    assert.doesNotMatch(written, /since it was loaded/);
});
test("training a loaded model at a different size asks before replacing it", async (t) => {
    const filePath = path.join(modelFolder(), "model.safetensors");
    await runConsole(t, ["2", filePath, "quit"]);
    const written = await runConsole(t, ["3", filePath, "1", "20", "n", "quit"]);
    assert.match(written, /needs a model 16 values wide, but the current model is 8 wide/);
    assert.match(written, /Training cancelled; the current model is unchanged\./);
});
/*
 * Loading GPT-2 small's weights (menu option 4). Loading the real 548 MB
 * file takes several seconds, so only one test does, and it is skipped if
 * the file has not been downloaded.
 */
test("loading GPT-2 small says how to download the weights when they are missing", async (t) => {
    const missing = path.join(modelFolder(), "gpt2-small.safetensors");
    const written = await runConsole(t, ["4", "quit"], undefined, missing);
    assert.match(written, /The GPT-2 small weights were not found at .*gpt2-small\.safetensors\. Download them with:/);
    assert.match(written, /curl\.exe -L https:\/\/huggingface\.co\/openai-community\/gpt2/);
    assert.doesNotMatch(written, /Loading /);
});
test("loading GPT-2 small from a file that is not GPT-2 reports the error and keeps the current model", async (t) => {
    const filePath = path.join(modelFolder(), "not-gpt2.safetensors");
    await runConsole(t, ["2", filePath, "quit"]);
    const written = await runConsole(t, ["4", "quit"], undefined, filePath);
    assert.match(written, /Loading GPT-2 small's weights from .*not-gpt2\.safetensors \(\d+\.\d MB\)/);
    assert.match(written, /Loading failed: .*has no array called wte\.weight/);
    assert.match(written, /The current model is unchanged\./);
});
/*
 * Chat is not tried here: each GPT-2 reply takes about a minute. The
 * generator itself is tested elsewhere.
 */
test("loading GPT-2 small replaces the model, showing progress, and sets longer replies", { skip: !fs.existsSync(ModelManager.GPT2_SMALL_FILE) && "the GPT-2 small weights have not been downloaded" }, async (t) => {
    const written = await runConsole(t, ["1", "1", "4", "y", "quit"]);
    assert.match(written, /trained 1 time since it was last saved, and its training will be lost if another model is loaded\.\nLoad anyway\?/);
    assert.match(written, /Loading GPT-2 small's weights from .*gpt2-small\.safetensors \(548\.1 MB\)/);
    assert.match(written, /Loaded: 10% 20% 30% 40% 50% 60% 70% 80% 90% 100%\n/);
    assert.match(written, /Model loaded in \d+\.\d seconds: 163\.0 million parameters, a model 768 values wide with 12 transformer blocks\./);
    assert.match(written, new RegExp(`Chat replies will be ${GPTConsole.GPT2_OUTPUT_LENGTH} tokens long and take about a minute each`));
    assert.match(written, /trained 0 times since it was loaded from gpt2-small\.safetensors/);
});
/*
 * Temperature and sampling method (menu options 5 and 6).
 */
test("temperature shows the current value and sets a new one", async (t) => {
    const written = await runConsole(t, ["5", "0.7", "5", "quit", "quit"]);
    assert.match(written, /The temperature is currently 1\./);
    assert.match(written, /The temperature is now 0\.7\./);
    // The second visit shows the new value, and "quit" leaves it unchanged.
    assert.match(written, /The temperature is currently 0\.7\./);
    assert.match(written, /The temperature is unchanged\./);
});
test("temperature asks again until the value is between 0 and 1", async (t) => {
    const written = await runConsole(t, ["5", "1.5", "-0.1", "warm", "0", "quit"]);
    assert.match(written, /"1\.5" is not a number between 0 and 1\./);
    assert.match(written, /"-0\.1" is not a number between 0 and 1\./);
    assert.match(written, /"warm" is not a number between 0 and 1\./);
    assert.equal(count(written, /New temperature\?/), 4);
    assert.match(written, /The temperature is now 0\./);
});
test("temperature notes that greedy sampling ignores it", async (t) => {
    const greedy = await runConsole(t, ["5", "", "quit"]);
    const topK = await runConsole(t, ["6", "top-k", "5", "", "quit"]);
    assert.match(greedy, /Greedy sampling ignores the temperature/);
    assert.doesNotMatch(topK, /Greedy sampling ignores the temperature/);
});
test("sampling method lists the methods and sets each one with its setting", async (t) => {
    const written = await runConsole(t, ["6", "top-k 30", "6", "TOP-P 0.8", "6", "greedy", "6", "", "quit"]);
    assert.match(written, /The sampling method is currently greedy\. The methods are:\n\s+greedy .*\n\s+top-p <p> .*\n\s+top-k <k> /);
    assert.match(written, /The sampling method is now top-k \(k = 30\)\./);
    assert.match(written, /The sampling method is currently top-k \(k = 30\)\./);
    assert.match(written, /The sampling method is now top-p \(p = 0\.8\)\./);
    assert.match(written, /The sampling method is now greedy\./);
    assert.match(written, /The sampling method is unchanged\./);
});
test("sampling method keeps the current setting when none is typed", async (t) => {
    const written = await runConsole(t, ["6", "top-k", "6", "top-p", "quit"]);
    assert.match(written, /The sampling method is now top-k \(k = 40\)\./);
    assert.match(written, /The sampling method is now top-p \(p = 0\.9\)\./);
});
test("sampling method asks again until the method and setting can be used", async (t) => {
    const written = await runConsole(t, [
        "6", "beam", "top-k 0", "top-k 2.5", "top-p 0", "top-p 1.5", "top-p high", "greedy 3", "top-k 30 40", "top-p 1", "quit",
    ]);
    assert.match(written, /"beam" is not a sampling method\./);
    assert.match(written, /"0" is not a whole number of tokens, 1 or more\./);
    assert.match(written, /"2\.5" is not a whole number of tokens, 1 or more\./);
    assert.match(written, /"0" is not a number above 0 and up to 1\./);
    assert.match(written, /"1\.5" is not a number above 0 and up to 1\./);
    assert.match(written, /"high" is not a number above 0 and up to 1\./);
    assert.match(written, /Greedy sampling does not take a setting\./);
    assert.match(written, /"top-k 30 40" has too many parts\./);
    assert.equal(count(written, /Please enter "greedy", "top-p <p>" or "top-k <k>", or "quit"\./), 8);
    assert.match(written, /The sampling method is now top-p \(p = 1\)\./);
});
test("chat replies with top-k and top-p sampling", async (t) => {
    const written = await runConsole(t, ["5", "0.8", "6", "top-k 5", "7", "Once upon a time", "quit", "6", "top-p 0.9", "7", "The cat", "quit", "quit"]);
    assert.equal(count(written, /Model: /), 2);
});
test("the temperature and sampling method are kept when another model is loaded", async (t) => {
    const filePath = path.join(modelFolder(), "decoding.safetensors");
    await runConsole(t, ["2", filePath, "quit"]);
    const written = await runConsole(t, ["5", "0.5", "6", "top-k 30", "3", filePath, "5", "", "6", "", "quit"]);
    assert.match(written, /Model loaded in/);
    assert.match(written, /Model loaded in[\s\S]*The temperature is currently 0\.5\./);
    assert.match(written, /Model loaded in[\s\S]*The sampling method is currently top-k \(k = 30\)\./);
});
test("the program ends when the input ends, even while asking for the hours or chatting", async (t) => {
    const whileAsking = await runConsole(t, ["1"]);
    const whileChatting = await runConsole(t, ["7", "Once upon a time"]);
    assert.match(whileAsking, /How many hours of training\?[\s\S]*Goodbye\./);
    assert.match(whileChatting, /Model: [\s\S]*Goodbye\./);
});
