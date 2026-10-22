import * as path from "path";
import * as readline from "readline";
import Hyperparameters from "./Hyperparameters.js";
import GPTModel from "./GPTModel.js";
import Generator from "./Generator.js";
import Trainer from "./Trainer.js";
import BPETokenizer from "./BPETokenizer.js";

/** Settings for a GPTConsole. Every one is optional. */
export interface GPTConsoleOptions {
  /**
   * Creates the hyperparameters for training for a given number of hours.
   * Defaults to Hyperparameters.forReducedTraining; tests pass a function
   * that builds a tiny model instead.
   */
  hyperparametersForHours?: (hours: number) => Hyperparameters;

  /** Path to the TinyStories training CSV file. Defaults to data/TinyStories_train.csv. */
  trainingFilePath?: string;

  /** Path to the TinyStories validation CSV file. Defaults to data/TinyStories_validation.csv. */
  validationFilePath?: string;

  /** Where the user's typing is read from. Defaults to the keyboard (process.stdin); tests pass their own stream. */
  input?: NodeJS.ReadableStream;

  /** Where the console writes its menus and replies. Defaults to the screen (process.stdout); tests pass their own stream. */
  output?: NodeJS.WritableStream;
}

/**
 * An interactive console for training the model and trying it out. run()
 * shows a menu with two options:
 *
 * 1. Train: asks how many hours to train for, then trains the model on
 *    TinyStories with Hyperparameters.forReducedTraining(hours).
 * 2. Chat: reads sentences typed by the user and shows how the model
 *    continues each one, until the user types "quit", which returns to the
 *    menu.
 *
 * Typing "quit" at the menu ends the program.
 *
 * The model, trainer and generator are shared, so chat always uses the
 * latest trained weights, and training again continues from where the last
 * training finished: the trainer keeps its optimizer, and each run trains on
 * the stories that follow the ones the last run used. The number of hours
 * can change the size of the model, though (forReducedTraining uses a wider
 * model for long training), and a model of a different size has to start
 * again from random weights; the console asks before throwing training away.
 * Before any training the model's weights are random, so its replies are
 * random tokens.
 *
 * The trained weights are kept in memory only, so they are lost when the
 * program ends.
 */
class GPTConsole {
  /** The keyword that leaves chat, or the program when typed at the menu. */
  static readonly QUIT: string = "quit";

  /**
   * The most hours of training the console accepts. A larger number is more
   * likely a typing mistake than a plan, and is far beyond the training
   * times that forReducedTraining's estimates were measured on.
   */
  static readonly MAXIMUM_HOURS: number = 100;

  /** The hours used to build the model before any training has been chosen, so that chat works straight away. */
  private static readonly INITIAL_HOURS: number = 2.5;

  /** Creates the hyperparameters for a number of hours of training. */
  private readonly hyperparametersForHours: (hours: number) => Hyperparameters;

  /*
   * The current model and the objects built with it. They are set by
   * buildModel(), which the constructor calls; the "!" tells TypeScript that
   * they are set, since it cannot see into buildModel().
   */

  /** The settings the current model was built with, shared by the trainer and the generator. */
  private hyperparameters!: Hyperparameters;

  /** The model being trained and chatted with. */
  private model!: GPTModel;

  /** Trains the model. */
  private trainer!: Trainer;

  /** Generates the model's continuation of each sentence typed in chat. */
  private generator!: Generator;

  /** Path to the TinyStories training CSV file. */
  private readonly trainingFilePath: string;

  /** Path to the TinyStories validation CSV file. */
  private readonly validationFilePath: string;

  /** Where the user's typing is read from. */
  private readonly input: NodeJS.ReadableStream;

  /** Where menus and replies are written. */
  private readonly output: NodeJS.WritableStream;

  /** The number of times the current model has been trained, shown in the menu and chat so the user knows what state it is in. */
  private trainingRuns: number = 0;

  /**
   * The user's typing, one line at a time. Reading lines from this iterator,
   * rather than with readline's question(), keeps every line even if it
   * arrives before the console has asked for it (for example, when input is
   * piped in or typed during training).
   */
  private lines: AsyncIterator<string> | null = null;

  /**
   * Creates a GPTConsole and builds an untrained model, so that chat can be
   * tried before training. The model is built from the hyperparameters for
   * INITIAL_HOURS (2.5) of training; choosing Train may rebuild it at a
   * different size.
   *
   * @param options - Optional settings; see GPTConsoleOptions.
   */
  constructor(options: GPTConsoleOptions = {}) {
    this.hyperparametersForHours = options.hyperparametersForHours ?? ((hours: number) => Hyperparameters.forReducedTraining(hours));

    // The data folder is found from this compiled file's location (dist/), so the console works however it is launched.
    const dataFolder: string = path.join(import.meta.dirname, "..", "data");
    this.trainingFilePath = options.trainingFilePath ?? path.join(dataFolder, "TinyStories_train.csv");
    this.validationFilePath = options.validationFilePath ?? path.join(dataFolder, "TinyStories_validation.csv");

    this.input = options.input ?? process.stdin;
    this.output = options.output ?? process.stdout;

    this.buildModel(this.hyperparametersForHours(GPTConsole.INITIAL_HOURS));
  }

  /**
   * Starts the interactive console: shows the menu, carries out each choice,
   * and shows the menu again, until the user types "quit" at the menu or the
   * input ends.
   *
   * @returns A promise that resolves when the console has ended.
   */
  async run(): Promise<void> {
    const lineReader = readline.createInterface({ input: this.input, terminal: false });
    this.lines = lineReader[Symbol.asyncIterator]();

    try {
      this.write("GPT console: train a small GPT model on TinyStories and chat with it.");
      let running: boolean = true;
      while (running) {
        this.write("");
        this.write(`Menu (the model has been trained ${this.trainingRuns} time${this.trainingRuns === 1 ? "" : "s"}):`);
        this.write("  1. Train");
        this.write("  2. Chat");
        const choice: string | null = await this.ask(`Choose 1 or 2 (or "${GPTConsole.QUIT}" to exit): `);

        // train() and chat() return false if the input ended while they were asking for something, which also ends the program.
        if (choice === null || choice.toLowerCase() === GPTConsole.QUIT) {
          running = false;
        } else if (choice === "1") {
          running = await this.train();
        } else if (choice === "2") {
          running = await this.chat();
        } else {
          this.write(`"${choice}" is not an option.`);
        }
      }
      this.write("Goodbye.");
    } finally {
      lineReader.close();
    }
  }

  /**
   * Menu option 1: asks how many hours to train for, works out the
   * hyperparameters for that time, and trains the model on TinyStories. The
   * trainer prints its progress as it goes. Training blocks the console until
   * it finishes. If the data cannot be loaded (for example, if a file is
   * missing), the error is shown and the console returns to the menu.
   *
   * @returns True to return to the menu; false if the input ended, which ends the program.
   */
  private async train(): Promise<boolean> {
    const hours: number | null | undefined = await this.askHours();
    if (hours === undefined) {
      return false;
    }
    if (hours === null) {
      this.write("Training cancelled.");
      return true;
    }

    const planned: Hyperparameters = this.hyperparametersForHours(hours);

    if (GPTConsole.sameArchitecture(planned, this.hyperparameters)) {
      // The current model is the right size: keep its weights, and train it with the new amount of data.
      GPTConsole.copyTrainingSettings(planned, this.hyperparameters);
    } else {
      // A model of a different size has to start again from random weights. Ask first if that would lose training.
      if (this.trainingRuns > 0) {
        this.write(
          `Training for ${hours} hours needs a model ${planned.embeddingSize} values wide, but the current model is ` +
          `${this.hyperparameters.embeddingSize} wide. A new model would start again from random weights, losing the current training.`
        );
        const answer: string | null = await this.ask("Start a new model? (y/n): ");
        if (answer === null) {
          return false;
        }
        if (answer.toLowerCase() !== "y" && answer.toLowerCase() !== "yes") {
          this.write("Training cancelled; the current model is unchanged.");
          return true;
        }
      }
      this.buildModel(planned);
    }

    this.writePlan(hours);
    try {
      this.trainer.loadTinyStories(this.trainingFilePath, this.validationFilePath);
      const losses = this.trainer.train();
      this.trainingRuns++;

      const last = losses[losses.length - 1];
      this.write(`Training finished. Training loss: ${last.trainingLoss.toFixed(3)}. Validation loss: ${last.validationLoss.toFixed(3)}.`);
    } catch (error) {
      this.write(`Training failed: ${error instanceof Error ? error.message : String(error)}`);
    }
    return true;
  }

  /**
   * Asks how many hours to train for, repeating the question until the
   * answer is sensible: a number greater than 0 and no more than
   * MAXIMUM_HOURS. An empty answer or "quit" cancels.
   *
   * @returns The hours; null if the user cancelled; undefined if the input ended.
   */
  private async askHours(): Promise<number | null | undefined> {
    for (;;) {
      const answer: string | null = await this.ask(
        `How many hours of training? (e.g. 2.5; press Enter or type "${GPTConsole.QUIT}" to cancel): `
      );
      if (answer === null) {
        return undefined;
      }
      if (answer === "" || answer.toLowerCase() === GPTConsole.QUIT) {
        return null;
      }

      /*
       * Number() rather than parseFloat(), because parseFloat accepts text
       * that merely starts with a number ("2 hours" gives 2), whereas Number()
       * rejects anything that is not entirely a number.
       */
      const hours: number = Number(answer);
      if (!Number.isFinite(hours)) {
        this.write(`"${answer}" is not a number of hours.`);
      } else if (hours <= 0) {
        this.write("The number of hours must be more than 0.");
      } else if (hours > GPTConsole.MAXIMUM_HOURS) {
        this.write(`The most the console accepts is ${GPTConsole.MAXIMUM_HOURS} hours.`);
      } else {
        return hours;
      }
    }
  }

  /**
   * Describes the training about to start: how many stories, how big the
   * model is, and roughly when training will finish.
   *
   * @param hours - The hours of training chosen.
   */
  private writePlan(hours: number): void {
    const finish: Date = new Date(Date.now() + hours * 60 * 60 * 1000);
    const finishTime: string = finish.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    const stories: number = this.hyperparameters.numberTrainingStories;
    const epochs: number = this.hyperparameters.numberEpochs;
    // Each run continues from where the trainer's last run finished in the training file.
    const firstStory: number = this.trainer.tinyStoriesLoaded + 1;

    this.write(
      `Training for about ${hours} hour${hours === 1 ? "" : "s"}: ${stories} stories ` +
      `(numbers ${firstStory} to ${firstStory + stories - 1}), ${epochs} epoch${epochs === 1 ? "" : "s"}, ` +
      `a model ${this.hyperparameters.embeddingSize} values wide with ` +
      `${(this.model.getParameterCount() / 1e6).toFixed(1)} million parameters.`
    );
    this.write(`Expected to finish at about ${finishTime}. The console is unavailable until then.`);
  }

  /**
   * Menu option 2: reads sentences from the user and shows the model's
   * continuation of each, until the user types "quit". Empty lines are
   * ignored.
   *
   * @returns True to return to the menu; false if the input ended, which ends the program.
   */
  private async chat(): Promise<boolean> {
    this.write(`Chat: type the start of a sentence and the model will continue it. Type "${GPTConsole.QUIT}" to return to the menu.`);
    if (this.trainingRuns === 0) {
      this.write("(The model has not been trained yet, so its replies will be random.)");
    }

    for (;;) {
      const sentence: string | null = await this.ask("You: ");
      if (sentence === null) {
        return false;
      }
      if (sentence.toLowerCase() === GPTConsole.QUIT) {
        return true;
      }
      if (sentence !== "") {
        this.write(`Model: ${this.reply(sentence)}`);
      }
    }
  }

  /**
   * Generates the model's continuation of a sentence. The generator returns
   * the sentence followed by the generated text; only the generated text is
   * returned here, since the user has just typed the sentence.
   *
   * TinyStories marks the end of each story with <|endoftext|>, so a trained
   * model often generates it when it finishes a story, and then starts a new,
   * unrelated one. The reply is cut off at that point.
   *
   * @param sentence - The text typed by the user.
   * @returns The model's continuation.
   */
  private reply(sentence: string): string {
    const continuation: string = this.generator.generate(sentence).slice(sentence.length);
    const endOfStory: number = continuation.indexOf(BPETokenizer.END_OF_TEXT);
    return endOfStory === -1 ? continuation : continuation.slice(0, endOfStory) + " [end of story]";
  }

  /**
   * Builds a new, untrained model from the given hyperparameters, with its
   * own trainer and generator, replacing the current ones.
   *
   * @param hyperparameters - The settings for the new model.
   */
  private buildModel(hyperparameters: Hyperparameters): void {
    this.hyperparameters = hyperparameters;
    // The generator needs generation mode; the trainer switches to training mode while it trains and then switches back.
    this.hyperparameters.mode = Hyperparameters.GENERATION_MODE;

    this.model = new GPTModel(this.hyperparameters);
    this.trainer = new Trainer(this.hyperparameters, this.model);
    this.generator = new Generator(this.hyperparameters, this.model);
    this.trainingRuns = 0;
  }

  /**
   * Checks whether two sets of hyperparameters build models of the same
   * shape, so that a model built from one can carry on training with the
   * other.
   *
   * @param a - One set of hyperparameters.
   * @param b - The other.
   * @returns True if every setting that decides the shape of the model's weights is the same.
   */
  private static sameArchitecture(a: Hyperparameters, b: Hyperparameters): boolean {
    return (
      a.embeddingSize === b.embeddingSize &&
      a.numberAttentionHeads === b.numberAttentionHeads &&
      a.numberTransformerBlocks === b.numberTransformerBlocks &&
      a.contextLength === b.contextLength &&
      a.qkvBias === b.qkvBias &&
      a.feedForwardBias === b.feedForwardBias
    );
  }

  /**
   * Copies the settings that control training, but not the shape of the
   * model, from one set of hyperparameters to another. The current model's
   * components keep a reference to their Hyperparameters, so the new values
   * take effect at the next training.
   *
   * @param from - The hyperparameters planned for the new training.
   * @param to - The hyperparameters the current model was built with.
   */
  private static copyTrainingSettings(from: Hyperparameters, to: Hyperparameters): void {
    to.numberTrainingStories = from.numberTrainingStories;
    to.numberValidationStories = from.numberValidationStories;
    to.numberEpochs = from.numberEpochs;
    to.batchSize = from.batchSize;
    to.learningRate = from.learningRate;
    to.outputLength = from.outputLength;
  }

  /**
   * Shows a prompt and waits for the user to type a line.
   *
   * @param prompt - The text shown before the user's typing, on the same line.
   * @returns The line typed, with surrounding spaces removed, or null if the input has ended.
   */
  private async ask(prompt: string): Promise<string | null> {
    this.output.write(prompt);
    const result = await this.lines!.next();
    if (result.done) {
      this.write("");
      return null;
    }
    return result.value.trim();
  }

  /**
   * Writes a line of text to the output.
   *
   * @param text - The text to write.
   */
  private write(text: string): void {
    this.output.write(text + "\n");
  }
}

export default GPTConsole;
