import { test } from "node:test";
import assert from "node:assert/strict";
import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import { PassThrough, Writable } from "stream";
import Hyperparameters from "../Hyperparameters.js";
import GPTConsole from "../GPTConsole.js";
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
 */
const runConsole = async (t, typed, trainingFilePath = writeStoriesCsv(STORIES)) => {
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
        input,
        output,
    });
    await gptConsole.run();
    return written;
};
/** Counts how many times a pattern appears in the text. */
const count = (text, pattern) => text.match(new RegExp(pattern.source, "g"))?.length ?? 0;
test("the menu offers Train and Chat, and quit ends the program", async (t) => {
    const written = await runConsole(t, ["quit"]);
    assert.match(written, /1\. Train/);
    assert.match(written, /2\. Chat/);
    assert.match(written, /Goodbye\./);
});
test("chat shows the model's continuation of each sentence, and quit returns to the menu", async (t) => {
    const written = await runConsole(t, ["2", "Once upon a time", "", "The cat", "quit", "quit"]);
    // Two replies: one per sentence, none for the empty line.
    assert.equal(count(written, /Model: /), 2);
    assert.match(written, /has not been trained yet/);
    // The menu is shown again after leaving chat (twice in all).
    assert.equal(count(written, /2\. Chat/), 2);
    assert.match(written, /Goodbye\./);
});
test("train asks for the hours, passes them on, describes the plan and trains", async (t) => {
    const written = await runConsole(t, ["1", "2.5", "2", "Once upon a time", "quit", "quit"]);
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
test("a training error is shown and the console returns to the menu", async (t) => {
    const written = await runConsole(t, ["1", "1", "quit"], path.join(os.tmpdir(), "no-such-folder", "missing.csv"));
    assert.match(written, /Training failed: .*no such file/);
    assert.match(written, /trained 0 times/);
    assert.match(written, /Goodbye\./);
});
test("quit is accepted in any case, and an unknown choice is reported", async (t) => {
    const written = await runConsole(t, ["3", "2", "QUIT", "Quit"]);
    assert.match(written, /"3" is not an option\./);
    assert.match(written, /Goodbye\./);
});
test("the program ends when the input ends, even while asking for the hours or chatting", async (t) => {
    const whileAsking = await runConsole(t, ["1"]);
    const whileChatting = await runConsole(t, ["2", "Once upon a time"]);
    assert.match(whileAsking, /How many hours of training\?[\s\S]*Goodbye\./);
    assert.match(whileChatting, /Model: [\s\S]*Goodbye\./);
});
