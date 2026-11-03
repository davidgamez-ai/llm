import { test, type TestContext } from "node:test";
import assert from "node:assert/strict";
import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import { PassThrough, Writable } from "stream";
import Hyperparameters from "../Hyperparameters.js";
import GPTConsole, { GPTConsoleOptions, ModelSize } from "../GPTConsole.js";
import TrainingPlanner from "../TrainingPlanner.js";
import ModelManager from "../ModelManager.js";

/** Writes a CSV file in the TinyStories format (a "text" column of quoted stories) and returns its path. */
const writeStoriesCsv = (stories: string[]): string => {
  const folder: string = fs.mkdtempSync(path.join(os.tmpdir(), "console-test-"));
  const filePath: string = path.join(folder, "stories.csv");
  fs.writeFileSync(filePath, "text\n" + stories.map((story) => `"${story}"`).join("\n") + "\n", "utf-8");
  return filePath;
};

// Enough for three runs of 10 stories each, since each run continues with new stories.
const STORIES: string[] = Array.from({ length: 30 }, (_, i) => `Once upon a time there was a cat called Tom ${i}. The end.`);

/** Writes text to a new file in a temporary folder and returns its path. */
const writeTextFile = (text: string): string => {
  const filePath: string = path.join(fs.mkdtempSync(path.join(os.tmpdir(), "console-test-")), "data.txt");
  fs.writeFileSync(filePath, text, "utf-8");
  return filePath;
};

// Longer than the least text training uses, whichever text it stands in for.
const TEXT: string = "The cat sat on the mat. The dog sat on the log. ".repeat(100);

/** The sizes each test's console was asked to build hyperparameters for, in order. */
let requestedSizes: ModelSize[] = [];

/**
 * A tiny stand-in for the model sizes, so the tests run quickly. Like the real
 * ones, the sizes have different widths: 8 values wide for small and 16 for
 * medium and large. It records the sizes asked for.
 */
const tinyHyperparametersForSize = (size: ModelSize): Hyperparameters => {
  requestedSizes.push(size);
  const hyperparameters = new Hyperparameters(size === "small" ? 8 : 16, 4, 0.1, Hyperparameters.TRAINING_MODE, 2, 1, 3);
  hyperparameters.seed = 42;
  hyperparameters.outputLength = 5;
  hyperparameters.numberValidationExamples = 2;
  return hyperparameters;
};

/**
 * Runs a GPTConsole with the tiny hyperparameters on the given lines of typed
 * input, starting from the question about the size of model, and returns
 * everything it wrote. The training speed is set so low that every training
 * run uses the fewest stories, 10. The trainer's own progress messages go to
 * console.log, which is silenced.
 *
 * @param t - The test context, used to silence console.log.
 * @param typed - The lines the user types, in order.
 * @param trainingFilePath - The training CSV. Defaults to a small file of STORIES.
 * @param gpt2FilePath - The GPT-2 small weights. Defaults to the console's own default, models/gpt2-small-weights.safetensors.
 * @param options - Any other console settings to use.
 */
const runConsoleFromStart = async (
  t: TestContext,
  typed: string[],
  trainingFilePath: string = writeStoriesCsv(STORIES),
  gpt2FilePath?: string,
  options: GPTConsoleOptions = {}
): Promise<string> => {
  t.mock.method(console, "log", () => {});
  requestedSizes = [];

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
    hyperparametersForSize: tinyHyperparametersForSize,
    parameterTokensPerHour: 1,
    trainingFilePath,
    validationFilePath: writeStoriesCsv(STORIES.slice(0, 2)),
    verdictFilePath: writeTextFile(TEXT),
    shakespeareFilePath: writeTextFile(TEXT),
    gpt2FilePath,
    input,
    output,
    ...options,
  });
  await gptConsole.run();
  return written;
};

/**
 * Runs a GPTConsole as runConsoleFromStart does, choosing the small model at
 * startup, so the lines typed start at the menu.
 */
const runConsole = (t: TestContext, typed: string[], trainingFilePath?: string, gpt2FilePath?: string, options?: GPTConsoleOptions): Promise<string> =>
  runConsoleFromStart(t, ["1", ...typed], trainingFilePath, gpt2FilePath, options);

/** Counts how many times a pattern appears in the text. */
const count = (text: string, pattern: RegExp): number => text.match(new RegExp(pattern.source, "g"))?.length ?? 0;

test("the menu offers Pretrain, Fine tune, Save model, Load model, Load GPT-2 small weights, Temperature, Sampling method and Chat, in that order, and quit ends the program", async (t) => {
  const written = await runConsole(t, ["quit"]);

  assert.match(written, /1\. Pretrain\n\s+2\. Fine tune\n\s+3\. Save model\n\s+4\. Load model\n\s+5\. Load GPT-2 small weights\n\s+6\. Temperature\n\s+7\. Sampling method\n\s+8\. Chat/);
  assert.match(written, /Goodbye\./);
});

test("chat shows the model's continuation of each sentence, and quit returns to the menu", async (t) => {
  const written = await runConsole(t, ["8", "1", "Once upon a time", "", "The cat", "quit", "quit"]);

  // Two replies: one per sentence, none for the empty line.
  assert.equal(count(written, /Model: /), 2);
  assert.match(written, /has not been trained yet/);
  // The menu is shown again after leaving chat (twice in all).
  assert.equal(count(written, /8\. Chat/), 2);
  assert.match(written, /Goodbye\./);
});

test("chat offers three modes, repeating the question until the answer is an option", async (t) => {
  const written = await runConsole(t, ["8", "0", "alpaca", "2", "quit", "quit"]);

  assert.match(written, /Chat modes:\n\s+1\. Standard\n\s+2\. Early stop\n\s+3\. Alpaca\n/);
  assert.match(written, /"0" is not an option\./);
  assert.match(written, /"alpaca" is not an option\./);
  assert.match(written, /Early stop chat:/);
});

test("pressing Enter or typing quit at the chat modes returns to the menu without chatting", async (t) => {
  for (const cancel of ["", "quit"]) {
    const written = await runConsole(t, ["8", cancel, "quit"]);

    assert.doesNotMatch(written, /You: /);
    assert.equal(count(written, /8\. Chat/), 2);
  }
});

test("the program ends if the input ends while asking for the chat mode", async (t) => {
  const written = await runConsole(t, ["8"]);

  assert.match(written, /Chat modes:[\s\S]*Goodbye\./);
});

test("each chat mode says what it does, using the limits of the model", async (t) => {
  const written = await runConsole(t, ["8", "1", "quit", "8", "2", "quit", "8", "3", "quit", "quit"]);

  // The tiny model's outputLength is 5, so the stopping modes use the larger limit.
  assert.match(written, /Standard chat: type the start of a sentence and the model will continue it for 5 tokens/);
  assert.match(written, new RegExp(`Early stop chat: [^\\n]*at most ${GPTConsole.STOPPING_REPLY_LENGTH} tokens`));
  assert.match(written, new RegExp(`Alpaca chat: type an instruction [^\\n]*at most ${GPTConsole.STOPPING_REPLY_LENGTH} tokens`));
});

test("the Alpaca chat mode warns about a prompt too long for the model's context, and the other modes do not", async (t) => {
  // The tiny model's context of 4 tokens is shorter than any Alpaca prompt.
  const alpaca = await runConsole(t, ["8", "3", "Name a colour.", "quit", "quit"]);
  const standard = await runConsole(t, ["8", "1", "Name a colour.", "quit", "quit"]);
  const earlyStop = await runConsole(t, ["8", "2", "Name a colour.", "quit", "quit"]);

  assert.match(alpaca, /\(The prompt is \d+ tokens and the model's context is 4, so the start of the template may be lost/);
  assert.match(alpaca, /Model: /);
  assert.doesNotMatch(standard, /start of the template/);
  assert.doesNotMatch(earlyStop, /start of the template/);
});

/*
 * What each chat mode really does is tested with a model trained to remember
 * one Alpaca example, in ChatModes.test.ts.
 */

test("the model is asked for at startup, repeating until the answer is an option, and a model of the size chosen is created", async (t) => {
  const written = await runConsoleFromStart(t, ["small", "0", "5", "", "2", "quit"]);

  assert.match(written, /Model:\n\s+1\. Small\n\s+2\. Medium\n\s+3\. Large \(GPT-2 small\)\n\s+4\. Load model\n/);
  assert.match(written, /"small" is not an option\./);
  assert.match(written, /"0" is not an option\./);
  assert.match(written, /"5" is not an option\./);
  assert.match(written, /"" is not an option\./);
  assert.equal(count(written, /Choose 1 to 4/), 5);
  assert.deepEqual(requestedSizes, ["medium"]);
  assert.match(written, /Created a model with random weights: \d+\.\d million parameters, 16 values wide with 1 transformer blocks\./);
  assert.match(written, /1\. Pretrain/);
});

test("each startup option creates the matching size of model", async (t) => {
  for (const [choice, size] of [["1", "small"], ["2", "medium"], ["3", "large"]]) {
    await runConsoleFromStart(t, [choice, "quit"]);
    assert.deepEqual(requestedSizes, [size]);
  }
});

test("the default sizes create models of the sizes the hyperparameters define", async (t) => {
  // Only the small model is built: it is the quickest, and the sizes' settings are tested with Hyperparameters.
  let written = "";
  const input = new PassThrough();
  input.end("1\nquit\n");
  const output = new Writable({
    write(chunk, _encoding, callback) {
      written += chunk.toString();
      callback();
    },
  });
  t.mock.method(console, "log", () => {});

  await new GPTConsole({ input, output }).run();

  assert.match(written, /Created a model with random weights: 13\.7 million parameters, 128 values wide with 4 transformer blocks\./);
});

test("typing quit at the startup question, or the input ending there, ends the program without showing the menu", async (t) => {
  for (const typed of [["quit"], ["QUIT"], []]) {
    const written = await runConsoleFromStart(t, typed);

    assert.doesNotMatch(written, /1\. Pretrain/);
    assert.deepEqual(requestedSizes, []);
    assert.match(written, /Goodbye\./);
  }
});

test("pretrain asks for the dataset and the hours, describes the plan and trains the current model", async (t) => {
  const written = await runConsole(t, ["1", "3", "2.5", "8", "1", "Once upon a time", "quit", "quit"]);

  // Only the model created at startup is built.
  assert.deepEqual(requestedSizes, ["small"]);
  assert.match(written, /Datasets for pre-training:\n\s+1\. The Verdict\n\s+2\. Tiny Shakespeare\n\s+3\. TinyStories\n\s+4\. Alpaca\n/);
  assert.match(written, /How many hours of pre-training\?/);
  assert.match(written, /Pre-training on TinyStories for about 2\.5 hours: 10 stories \(numbers 1 to 10\), 1 epoch, batch size 2, learning rate 0\.001, windows of up to 4 tokens, a model 8 values wide/);
  assert.match(written, /Expected to finish at about/);
  assert.match(written, /Training finished\. Training loss: \d+\.\d{3}\. Validation loss: \d+\.\d{3}\./);
  assert.match(written, /trained 1 time\b/);
  assert.doesNotMatch(written, /has not been trained yet/);
  assert.match(written, /Model: /);
});

test("fine tune asks for the dataset and the hours, and uses the fine tuning learning rate and batch size", async (t) => {
  const filePath = path.join(modelFolder(), "trained.safetensors");
  await runConsole(t, ["1", "3", "1", "3", filePath, "quit"]);
  const written = await runConsole(t, ["4", filePath, "2", "3", "1", "quit", "y"]);

  assert.match(written, /Datasets for fine tuning:\n\s+1\. The Verdict\n/);
  assert.match(written, /How many hours of fine tuning\?/);
  assert.match(written, new RegExp(`Fine tuning on TinyStories for about 1 hour: 10 stories \\(numbers 1 to 10\\), 1 epoch, batch size 8, learning rate ${TrainingPlanner.FINE_TUNING_LEARNING_RATE},`));
  assert.match(written, /Training finished/);
  assert.doesNotMatch(written, /random weights, and fine tuning/);
});

test("fine tuning a model with random weights notes that it is meant for trained weights, but is allowed", async (t) => {
  const written = await runConsole(t, ["2", "3", "1", "quit", "y"]);

  assert.match(written, /The model has random weights, and fine tuning is meant for weights that have been trained/);
  assert.match(written, /Training finished/);
});

test("pre-training after fine tuning goes back to the pre-training learning rate and batch size", async (t) => {
  const written = await runConsole(t, ["2", "3", "1", "1", "3", "1", "quit", "y"]);

  assert.match(written, /Fine tuning on TinyStories[^\n]*batch size 8, learning rate 0\.00005[\s\S]*Pre-training on TinyStories[^\n]*batch size 2, learning rate 0\.001,/);
});

test("the text datasets can be pre-trained and fine tuned on, using only the start of the text when time is short", async (t) => {
  const written = await runConsole(t, ["1", "1", "1", "1", "2", "1", "2", "1", "1", "quit", "y"]);

  assert.match(written, /Pre-training on The Verdict for about 1 hour: the first \d+ characters, 1 epoch, batch size 2,/);
  assert.match(written, /Pre-training on Tiny Shakespeare for about 1 hour: the first \d+ characters, 1 epoch, batch size 2,/);
  assert.match(written, /Fine tuning on The Verdict for about 1 hour: the first \d+ characters, 1 epoch, batch size 8,/);
  assert.equal(count(written, /Training finished/), 3);
});

test("training for longer than a text can use says that it will finish early", async (t) => {
  // A very short text, so that its 50 epochs are quick, but with enough validation text for one window.
  const written = await runConsole(t, ["1", "1", "100", "quit", "y"], undefined, undefined, {
    parameterTokensPerHour: 1e11,
    verdictFilePath: writeTextFile("The cat sat on the mat. ".repeat(10)),
  });

  assert.match(written, /Pre-training on The Verdict for about 100 hours: the whole text, 50 epochs,/);
  assert.match(written, /There is only enough data for this model to train for about \d+\.\d hours, so it will finish early\./);
  assert.match(written, /Training finished/);
});

test("Alpaca can be trained on, with a model whose context is long enough for its examples", async (t) => {
  const examples = Array.from({ length: 12 }, (_, i) => ({ instruction: `Name item ${i}.`, input: "", output: `It is item ${i}.` }));
  const written = await runConsole(t, ["1", "4", "1", "quit", "y"], undefined, undefined, {
    alpacaFilePath: writeTextFile(JSON.stringify(examples)),
    hyperparametersForSize: () => {
      const hyperparameters = new Hyperparameters(8, 64, 0.1, Hyperparameters.TRAINING_MODE, 2, 1);
      hyperparameters.seed = 42;
      hyperparameters.numberValidationExamples = 2;
      return hyperparameters;
    },
  });

  assert.match(written, /Pre-training on Alpaca for about 1 hour: 10 examples, 1 epoch, batch size 2,/);
  assert.match(written, /Training finished/);
});

test("the dataset is asked for again until it is an option, and pressing Enter or typing quit cancels", async (t) => {
  const written = await runConsole(t, ["1", "0", "5", "Alpaca", "", "2", "quit", "quit"]);

  assert.match(written, /"0" is not an option\./);
  assert.match(written, /"5" is not an option\./);
  assert.match(written, /"Alpaca" is not an option\./);
  assert.match(written, /Pre-training cancelled\./);
  assert.match(written, /Fine tuning cancelled\./);
  assert.doesNotMatch(written, /How many hours/);
  assert.match(written, /trained 0 times/);
});

test("a dataset whose files are missing is reported before the hours are asked for", async (t) => {
  const written = await runConsole(t, ["1", "1", "quit"], undefined, undefined, { verdictFilePath: path.join(modelFolder(), "missing.txt") });

  assert.match(written, /The data for this dataset was not found at .*missing\.txt\./);
  assert.doesNotMatch(written, /How many hours/);
  assert.match(written, /trained 0 times/);
});

test("the hours are asked for again until they are a number greater than 0 and no more than 100", async (t) => {
  const written = await runConsole(t, ["1", "3", "abc", "2 hours", "0", "-3", "101", "Infinity", "1", "quit"]);

  assert.match(written, /"abc" is not a number of hours\./);
  assert.match(written, /"2 hours" is not a number of hours\./);
  assert.equal(count(written, /must be more than 0/), 2);
  assert.match(written, /The most the console accepts is 100 hours\./);
  assert.match(written, /"Infinity" is not a number of hours\./);
  // Asked seven times; trained once, with the first sensible answer.
  assert.equal(count(written, /How many hours of pre-training\?/), 7);
  assert.match(written, /Pre-training on TinyStories for about 1 hour:/);
  assert.match(written, /trained 1 time\b/);
});

test("pressing Enter or typing quit when asked for the hours cancels training", async (t) => {
  const written = await runConsole(t, ["1", "3", "", "1", "3", "QUIT", "quit"]);

  assert.equal(count(written, /Pre-training cancelled\./), 2);
  assert.doesNotMatch(written, /Training finished/);
  assert.match(written, /trained 0 times/);
  assert.match(written, /Goodbye\./);
});

test("training again continues training the same model, on the next stories", async (t) => {
  const written = await runConsole(t, ["1", "3", "1", "1", "3", "3", "1", "3", "2", "quit"]);

  assert.equal(count(written, /Training finished/), 3);
  assert.match(written, /trained 3 times/);
  // Each run starts where the last one finished, with no new model.
  assert.match(written, /numbers 1 to 10\)[\s\S]*numbers 11 to 20\)[\s\S]*numbers 21 to 30\)/);
  assert.deepEqual(requestedSizes, ["small"]);
});

test("train uses the size of model chosen at startup", async (t) => {
  const written = await runConsoleFromStart(t, ["2", "1", "3", "20", "quit", "y"]);

  assert.match(written, /Pre-training on TinyStories for about 20 hours: 10 stories \(numbers 1 to 10\), 1 epoch, [^\n]*a model 16 values wide/);
  assert.match(written, /Training finished/);
});

test("quitting with an untrained model does not ask for confirmation", async (t) => {
  const written = await runConsole(t, ["8", "1", "Hello", "quit", "quit"]);

  assert.doesNotMatch(written, /Quit anyway\?/);
  assert.match(written, /Goodbye\./);
});

test("quitting with a trained model asks for confirmation; 'n' stays in the program and 'y' quits", async (t) => {
  const written = await runConsole(t, ["1", "3", "1", "quit", "n", "quit", "Y"]);

  assert.equal(count(written, /Quit anyway\? \(y\/n\)/), 2);
  assert.match(written, /The model has been trained 1 time, and its training will be lost when the program ends\./);
  // After "n" the menu is shown again: once at the start, once after training, once after declining.
  assert.equal(count(written, /8\. Chat/), 3);
  assert.equal(count(written, /Goodbye\./), 1);
  assert.match(written, /Quit anyway\? \(y\/n\): Goodbye\.\s*$/);
});

test("the program ends if the input ends while asking whether to quit", async (t) => {
  const written = await runConsole(t, ["1", "3", "1", "quit"]);

  assert.match(written, /Quit anyway\?[\s\S]*Goodbye\./);
});

test("a training error is shown and the console returns to the menu", async (t) => {
  const written = await runConsole(t, ["1", "3", "1", "quit"], writeTextFile("not_the_text_column\nA story.\n"));

  assert.match(written, /Training failed: /);
  assert.match(written, /trained 0 times/);
  assert.match(written, /Goodbye\./);
});

test("quit is accepted in any case, and an unknown choice is reported", async (t) => {
  const written = await runConsole(t, ["9", "3", "", "QUIT", "Quit"]);

  assert.match(written, /"9" is not an option\./);
  assert.match(written, /Goodbye\./);
});

/*
 * Saving and loading models (menu options 3 and 4).
 */

/** Creates an empty temporary folder for model files and returns its path. */
const modelFolder = (): string => fs.mkdtempSync(path.join(os.tmpdir(), "console-model-test-"));

test("save asks for a file name, saves the model with its progress, and quitting then does not ask for confirmation", async (t) => {
  const filePath = path.join(modelFolder(), "trained.safetensors");
  const written = await runConsole(t, ["1", "3", "1", "3", filePath, "quit"]);

  assert.match(written, /File to save the model to\?/);
  assert.match(written, /Saving the model \(\d+\.\d million parameters, \d+\.\d MB\) to /);
  assert.match(written, /Saved: 10% 20% 30% 40% 50% 60% 70% 80% 90% 100%\n/);
  assert.match(written, /Model saved in \d+\.\d seconds\./);
  assert.ok(fs.statSync(filePath).size > 0);
  assert.doesNotMatch(written, /Quit anyway\?/);
});

test("save adds .safetensors to a file name without an extension, and removes surrounding quotes", async (t) => {
  const folder = modelFolder();
  await runConsole(t, ["3", `"${path.join(folder, "model")}"`, "quit"]);

  assert.ok(fs.existsSync(path.join(folder, "model.safetensors")));
});

test("save asks before overwriting a file; 'n' asks for another name and leaves the file unchanged, 'y' overwrites", async (t) => {
  const folder = modelFolder();
  const existing = path.join(folder, "existing.safetensors");
  fs.writeFileSync(existing, "keep me");
  const other = path.join(folder, "other.safetensors");

  const declined = await runConsole(t, ["3", existing, "n", other, "quit"]);
  assert.match(declined, /A file already exists at .*existing\.safetensors\. Overwrite it\? \(y\/n\)/);
  assert.match(declined, /The existing file has not been changed\./);
  assert.equal(fs.readFileSync(existing, "utf-8"), "keep me");
  assert.ok(fs.existsSync(other));

  const accepted = await runConsole(t, ["3", existing, "y", "quit"]);
  assert.match(accepted, /Model saved/);
  assert.notEqual(fs.readFileSync(existing, "utf-8"), "keep me");
});

test("save asks again for a file name in a folder that does not exist, a folder name, or a name with invalid characters", async (t) => {
  const folder = modelFolder();
  const written = await runConsole(t, [
    "3",
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
  const written = await runConsole(t, ["3", "", "4", "quit", "quit"]);

  assert.match(written, /Save cancelled\./);
  assert.match(written, /Load cancelled\./);
});

test("load replaces the model with a saved one, showing its progress, and chat uses it", async (t) => {
  const filePath = path.join(modelFolder(), "trained.safetensors");
  // Train and save, then start a new console and load the saved model.
  await runConsole(t, ["1", "3", "1", "3", filePath, "quit"]);
  const written = await runConsole(t, ["4", filePath, "8", "1", "Once upon a time", "quit", "quit"]);

  assert.match(written, /File to load the model from\?/);
  assert.match(written, /Loading the model from .*trained\.safetensors \(\d+\.\d MB\)/);
  assert.match(written, /Loaded: 10% 20% 30% 40% 50% 60% 70% 80% 90% 100%\n/);
  assert.match(written, /Model loaded in \d+\.\d seconds: \d+\.\d million parameters, a model 8 values wide with 1 transformer blocks\./);
  assert.match(written, /trained 0 times since it was loaded from trained\.safetensors/);
  assert.doesNotMatch(written, /has not been trained yet/);
  assert.match(written, /Model: /);
});

test("loading at startup uses the saved model without building another, and it can be chatted with and trained", async (t) => {
  const filePath = path.join(modelFolder(), "startup.safetensors");
  await runConsole(t, ["1", "3", "1", "3", filePath, "quit"]);

  const written = await runConsoleFromStart(t, ["4", filePath, "8", "1", "Once upon a time", "quit", "1", "3", "1", "quit", "y"]);

  assert.deepEqual(requestedSizes, []);
  assert.doesNotMatch(written, /Created a model with random weights/);
  assert.match(written, /Model loaded in \d+\.\d seconds: \d+\.\d million parameters, a model 8 values wide with 1 transformer blocks\./);
  assert.match(written, /trained 0 times since it was loaded from startup\.safetensors/);
  assert.doesNotMatch(written, /has not been trained yet/);
  assert.match(written, /Model: /);
  assert.match(written, /trained 1 time since it was loaded from startup\.safetensors/);
});

test("cancelling a load at startup asks again, and a model can then be created", async (t) => {
  for (const cancel of ["", "quit"]) {
    const written = await runConsoleFromStart(t, ["4", cancel, "1", "quit"]);

    assert.match(written, /Load cancelled\./);
    assert.equal(count(written, /Choose 1 to 4/), 2);
    assert.deepEqual(requestedSizes, ["small"]);
    assert.match(written, /1\. Pretrain/);
  }
});

test("a load that fails at startup shows the error and asks again, without opening the menu or mentioning a current model", async (t) => {
  const notAModel = writeTextFile("This is not a saved model.");
  const written = await runConsoleFromStart(t, ["4", notAModel, "quit"]);

  assert.match(written, /Loading failed: /);
  assert.doesNotMatch(written, /current model/);
  assert.equal(count(written, /Choose 1 to 4/), 2);
  assert.doesNotMatch(written, /1\. Pretrain/);
  assert.deepEqual(requestedSizes, []);
  assert.match(written, /Goodbye\./);
});

test("a file name with nothing there is asked for again at startup, and the program ends if the input ends", async (t) => {
  const missing = path.join(modelFolder(), "missing.safetensors");
  const written = await runConsoleFromStart(t, ["4", missing]);

  assert.equal(count(written, /File to load the model from\?/), 2);
  assert.doesNotMatch(written, /1\. Pretrain/);
  assert.match(written, /Goodbye\./);
});

test("load asks again for a file name where there is no file, or a folder", async (t) => {
  const folder = modelFolder();
  const filePath = path.join(folder, "model.safetensors");
  await runConsole(t, ["3", filePath, "quit"]);
  const written = await runConsole(t, ["4", path.join(folder, "missing.safetensors"), folder, filePath, "quit"]);

  assert.match(written, /There is no file at .*missing\.safetensors\. Please enter a different file name\./);
  assert.match(written, /is a folder, not a file\. Please enter a different file name\./);
  assert.match(written, /Model loaded/);
});

test("load asks before losing unsaved training; 'n' keeps the current model", async (t) => {
  const filePath = path.join(modelFolder(), "model.safetensors");
  await runConsole(t, ["3", filePath, "quit"]);
  const written = await runConsole(t, ["1", "3", "1", "4", filePath, "n", "quit", "y"]);

  assert.match(written, /trained 1 time since it was last saved, and its training will be lost if another model is loaded\./);
  assert.match(written, /Load cancelled; the current model is unchanged\./);
  assert.doesNotMatch(written, /Model loaded/);
  // Still unsaved, so quitting asks.
  assert.match(written, /Quit anyway\?/);
});

test("a file that is not a saved model is reported, and the current model is kept", async (t) => {
  const filePath = path.join(modelFolder(), "notes.txt");
  fs.writeFileSync(filePath, "not a model");
  const written = await runConsole(t, ["4", filePath, "quit"]);

  assert.match(written, /Loaded:\n/);
  assert.match(written, /Loading failed: .*not a valid safetensors file/);
  assert.match(written, /The current model is unchanged\./);
  assert.doesNotMatch(written, /since it was loaded/);
});

test("training a loaded model of another size keeps that model", async (t) => {
  const filePath = path.join(modelFolder(), "model.safetensors");
  // Save a medium model, then load it into a console that started with a small one.
  await runConsoleFromStart(t, ["2", "3", filePath, "quit"]);
  const written = await runConsole(t, ["4", filePath, "1", "3", "1", "quit", "y"]);

  assert.match(written, /Pre-training on TinyStories for about 1 hour: 10 stories \(numbers 1 to 10\), 1 epoch, [^\n]*a model 16 values wide/);
  assert.match(written, /Training finished/);
  assert.match(written, /trained 1 time since it was loaded from model\.safetensors/);
});

/*
 * Loading GPT-2 small's weights (menu option 5). Loading the real 548 MB
 * file takes several seconds, so only one test does, and it is skipped if
 * the file has not been downloaded.
 */

test("loading GPT-2 small says how to download the weights when they are missing", async (t) => {
  const missing = path.join(modelFolder(), "gpt2-small-weights.safetensors");
  const written = await runConsole(t, ["5", "quit"], undefined, missing);

  assert.match(written, /The GPT-2 small weights were not found at .*gpt2-small-weights\.safetensors\. Download them with:/);
  assert.match(written, /curl\.exe -L https:\/\/huggingface\.co\/openai-community\/gpt2/);
  assert.doesNotMatch(written, /Loading /);
});

test("loading GPT-2 small from a file that is not GPT-2 reports the error and keeps the current model", async (t) => {
  const filePath = path.join(modelFolder(), "not-gpt2.safetensors");
  await runConsole(t, ["3", filePath, "quit"]);
  const written = await runConsole(t, ["5", "quit"], undefined, filePath);

  assert.match(written, /Loading GPT-2 small's weights from .*not-gpt2\.safetensors \(\d+\.\d MB\)/);
  assert.match(written, /Loading failed: .*has no array called wte\.weight/);
  assert.match(written, /The current model is unchanged\./);
  assert.doesNotMatch(written, /has been replaced with a large model/);
});

/*
 * Chat is not tried here: each GPT-2 reply takes about a minute. The
 * generator itself is tested elsewhere.
 */
const GPT2_MISSING: string | false = !fs.existsSync(ModelManager.GPT2_SMALL_FILE) && "the GPT-2 small weights have not been downloaded";

test("loading GPT-2 small into a large model loads the weights into it, without replacing it", { skip: GPT2_MISSING }, async (t) => {
  t.mock.method(console, "log", () => {});
  let written = "";
  const input = new PassThrough();
  input.end("3\n5\nquit\n");
  const output = new Writable({
    write(chunk, _encoding, callback) {
      written += chunk.toString();
      callback();
    },
  });

  await new GPTConsole({ input, output }).run();

  assert.match(written, /The current model already has GPT-2 small's shape, so the weights were loaded into it\./);
  assert.doesNotMatch(written, /has been replaced with a large model/);
  assert.match(written, /Model loaded in \d+\.\d seconds: 163\.0 million parameters/);
  assert.match(written, /trained 0 times since it was loaded from gpt2-small-weights\.safetensors/);
});

test("loading GPT-2 small replaces a smaller model with a large one, says so, shows progress and sets longer replies", { skip: GPT2_MISSING }, async (t) => {
  const written = await runConsole(t, ["1", "3", "1", "5", "y", "quit"]);

  assert.match(written, /The current model \(8 values wide with 1 transformer blocks\) cannot hold GPT-2 small's weights, so it has been replaced with a large model\./);

  assert.match(written, /trained 1 time since it was last saved, and its training will be lost if another model is loaded\.\nLoad anyway\?/);
  assert.match(written, /Loading GPT-2 small's weights from .*gpt2-small-weights\.safetensors \(548\.1 MB\)/);
  assert.match(written, /Loaded: 10% 20% 30% 40% 50% 60% 70% 80% 90% 100%\n/);
  assert.match(written, /Model loaded in \d+\.\d seconds: 163\.0 million parameters, a model 768 values wide with 12 transformer blocks\./);
  assert.match(written, new RegExp(`Standard chat replies will be ${GPTConsole.GPT2_OUTPUT_LENGTH} tokens long and take about a minute each`));
  assert.match(written, new RegExp(`Early stop and Alpaca replies can run to ${GPTConsole.STOPPING_REPLY_LENGTH} tokens`));
  assert.match(written, /trained 0 times since it was loaded from gpt2-small-weights\.safetensors/);
});

/*
 * Temperature and sampling method (menu options 6 and 7).
 */

test("temperature shows the current value and sets a new one", async (t) => {
  const written = await runConsole(t, ["6", "0.7", "6", "quit", "quit"]);

  assert.match(written, /The temperature is currently 1\./);
  assert.match(written, /The temperature is now 0\.7\./);
  // The second visit shows the new value, and "quit" leaves it unchanged.
  assert.match(written, /The temperature is currently 0\.7\./);
  assert.match(written, /The temperature is unchanged\./);
});

test("temperature asks again until the value is between 0 and 1", async (t) => {
  const written = await runConsole(t, ["6", "1.5", "-0.1", "warm", "0", "quit"]);

  assert.match(written, /"1\.5" is not a number between 0 and 1\./);
  assert.match(written, /"-0\.1" is not a number between 0 and 1\./);
  assert.match(written, /"warm" is not a number between 0 and 1\./);
  assert.equal(count(written, /New temperature\?/), 4);
  assert.match(written, /The temperature is now 0\./);
});

test("temperature notes that greedy sampling ignores it", async (t) => {
  const greedy = await runConsole(t, ["6", "", "quit"]);
  const topK = await runConsole(t, ["7", "top-k", "6", "", "quit"]);

  assert.match(greedy, /Greedy sampling ignores the temperature/);
  assert.doesNotMatch(topK, /Greedy sampling ignores the temperature/);
});

test("sampling method lists the methods and sets each one with its setting", async (t) => {
  const written = await runConsole(t, ["7", "top-k 30", "7", "TOP-P 0.8", "7", "greedy", "7", "", "quit"]);

  assert.match(written, /The sampling method is currently greedy\. The methods are:\n\s+greedy .*\n\s+top-p <p> .*\n\s+top-k <k> /);
  assert.match(written, /The sampling method is now top-k \(k = 30\)\./);
  assert.match(written, /The sampling method is currently top-k \(k = 30\)\./);
  assert.match(written, /The sampling method is now top-p \(p = 0\.8\)\./);
  assert.match(written, /The sampling method is now greedy\./);
  assert.match(written, /The sampling method is unchanged\./);
});

test("sampling method keeps the current setting when none is typed", async (t) => {
  const written = await runConsole(t, ["7", "top-k", "7", "top-p", "quit"]);

  assert.match(written, /The sampling method is now top-k \(k = 40\)\./);
  assert.match(written, /The sampling method is now top-p \(p = 0\.9\)\./);
});

test("sampling method asks again until the method and setting can be used", async (t) => {
  const written = await runConsole(t, [
    "7", "beam", "top-k 0", "top-k 2.5", "top-p 0", "top-p 1.5", "top-p high", "greedy 3", "top-k 30 40", "top-p 1", "quit",
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
  const written = await runConsole(t, ["6", "0.8", "7", "top-k 5", "8", "1", "Once upon a time", "quit", "7", "top-p 0.9", "8", "1", "The cat", "quit", "quit"]);

  assert.equal(count(written, /Model: /), 2);
});

test("the temperature and sampling method are kept when another model is loaded", async (t) => {
  const filePath = path.join(modelFolder(), "decoding.safetensors");
  await runConsole(t, ["3", filePath, "quit"]);
  const written = await runConsole(t, ["6", "0.5", "7", "top-k 30", "4", filePath, "6", "", "7", "", "quit"]);

  assert.match(written, /Model loaded in/);
  assert.match(written, /Model loaded in[\s\S]*The temperature is currently 0\.5\./);
  assert.match(written, /Model loaded in[\s\S]*The sampling method is currently top-k \(k = 30\)\./);
});

test("the program ends when the input ends, even while asking for the dataset or the hours, or chatting", async (t) => {
  const whileAskingDataset = await runConsole(t, ["1"]);
  const whileAsking = await runConsole(t, ["1", "3"]);
  const whileChatting = await runConsole(t, ["8", "1", "Once upon a time"]);

  assert.match(whileAskingDataset, /Datasets for pre-training:[\s\S]*Goodbye\./);
  assert.match(whileAsking, /How many hours of pre-training\?[\s\S]*Goodbye\./);
  assert.match(whileChatting, /Model: [\s\S]*Goodbye\./);
});
