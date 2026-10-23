import * as fs from "fs";
import * as path from "path";
import * as readline from "readline";
import Hyperparameters from "./Hyperparameters.js";
import GPTModel from "./GPTModel.js";
import Generator from "./Generator.js";
import Trainer from "./Trainer.js";
import BPETokenizer from "./BPETokenizer.js";
import ModelManager from "./ModelManager.js";
/**
 * An interactive console for training the model and trying it out. run()
 * shows a menu with seven options:
 *
 * 1. Train: asks how many hours to train for, then trains the model on
 *    TinyStories with Hyperparameters.forReducedTraining(hours).
 * 2. Save model: asks for a file name and saves the model's weights and
 *    hyperparameters there (see ModelManager), asking before overwriting an
 *    existing file.
 * 3. Load model: asks for a file name and replaces the current model with
 *    the one saved there, asking first if the current model has training
 *    that has not been saved.
 * 4. Load GPT-2 small weights: replaces the current model with GPT-2 small,
 *    with OpenAI's pretrained weights (see ModelManager.loadGPT2), asking
 *    first in the same way as Load model.
 * 5. Temperature: shows the temperature used when generating text and asks
 *    for a new one, between 0 and 1.
 * 6. Sampling method: shows how the next token is chosen when generating
 *    text and asks for a new method: "greedy", "top-p" followed by p, or
 *    "top-k" followed by k (e.g. "top-k 30").
 * 7. Chat: reads sentences typed by the user and shows how the model
 *    continues each one, until the user types "quit", which returns to the
 *    menu.
 *
 * The temperature and sampling method are kept when the model is replaced
 * by training a model of a different size or by loading one.
 *
 * Typing "quit" at the menu ends the program, after asking for confirmation
 * if the model has training that has not been saved, since it would be lost.
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
 * A loaded model gets a new trainer, so training it again starts a new
 * optimizer and starts again from the first story.
 *
 * File names typed for saving and loading are relative to the folder the
 * program was started in, unless they are absolute; ".safetensors" is added
 * if no extension is given.
 */
class GPTConsole {
    /** The keyword that leaves chat, or the program when typed at the menu. */
    static QUIT = "quit";
    /**
     * The most hours of training the console accepts. A larger number is more
     * likely a typing mistake than a plan, and is far beyond the training
     * times that forReducedTraining's estimates were measured on.
     */
    static MAXIMUM_HOURS = 100;
    /** The hours used to build the model before any training has been chosen, so that chat works straight away. */
    static INITIAL_HOURS = 2.5;
    /**
     * The number of tokens GPT-2 small generates in each chat reply: enough
     * for a sentence or two. The default Hyperparameters it is built with only
     * generate 2. GPT-2 small takes over a second per token in this
     * implementation, and each token takes a little longer than the last, since
     * the whole sequence is run through the model again for every token (about
     * 13 seconds for 10 tokens and 70 for 30), so 25 tokens take about a minute.
     */
    static GPT2_OUTPUT_LENGTH = 15;
    /** The extension added to file names typed without one when saving or loading. */
    static MODEL_EXTENSION = ".safetensors";
    /** Characters Windows does not allow in file names (and control characters, which no system allows sensibly). */
    static INVALID_FILE_NAME_CHARACTERS = /[<>:"|?*\x00-\x1f]/;
    /** Saves and loads the model's weights. */
    modelManager = new ModelManager();
    /** The file the current model was loaded from, or null if it was built here. */
    loadedFrom = null;
    /** True if the model has been trained since it was built, loaded or last saved, so quitting or loading would lose training. */
    unsavedTraining = false;
    /** Creates the hyperparameters for a number of hours of training. */
    hyperparametersForHours;
    /*
     * The current model and the objects built with it. They are set by
     * buildModel(), which the constructor calls; the "!" tells TypeScript that
     * they are set, since it cannot see into buildModel().
     */
    /** The settings the current model was built with, shared by the trainer and the generator. */
    hyperparameters;
    /** The model being trained and chatted with. */
    model;
    /** Trains the model. */
    trainer;
    /** Generates the model's continuation of each sentence typed in chat. */
    generator;
    /** Path to the TinyStories training CSV file. */
    trainingFilePath;
    /** Path to the TinyStories validation CSV file. */
    validationFilePath;
    /** Path to OpenAI's GPT-2 small weights. */
    gpt2FilePath;
    /** Where the user's typing is read from. */
    input;
    /** Where menus and replies are written. */
    output;
    /** The number of times the current model has been trained, shown in the menu and chat so the user knows what state it is in. */
    trainingRuns = 0;
    /**
     * The user's typing, one line at a time. Reading lines from this iterator,
     * rather than with readline's question(), keeps every line even if it
     * arrives before the console has asked for it (for example, when input is
     * piped in or typed during training).
     */
    lines = null;
    /**
     * Creates a GPTConsole and builds an untrained model, so that chat can be
     * tried before training. The model is built from the hyperparameters for
     * INITIAL_HOURS (2.5) of training; choosing Train may rebuild it at a
     * different size.
     *
     * @param options - Optional settings; see GPTConsoleOptions.
     */
    constructor(options = {}) {
        this.hyperparametersForHours = options.hyperparametersForHours ?? ((hours) => Hyperparameters.forReducedTraining(hours));
        // The data folder is found from this compiled file's location (dist/), so the console works however it is launched.
        const dataFolder = path.join(import.meta.dirname, "..", "data");
        this.trainingFilePath = options.trainingFilePath ?? path.join(dataFolder, "TinyStories_train.csv");
        this.validationFilePath = options.validationFilePath ?? path.join(dataFolder, "TinyStories_validation.csv");
        this.gpt2FilePath = options.gpt2FilePath ?? ModelManager.GPT2_SMALL_FILE;
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
    async run() {
        const lineReader = readline.createInterface({ input: this.input, terminal: false });
        this.lines = lineReader[Symbol.asyncIterator]();
        try {
            this.write("GPT console: train a small GPT model on TinyStories and chat with it.");
            let running = true;
            while (running) {
                this.write("");
                const loaded = this.loadedFrom === null ? "" : ` since it was loaded from ${path.basename(this.loadedFrom)}`;
                this.write(`Menu (the model has been trained ${this.trainingRuns} time${this.trainingRuns === 1 ? "" : "s"}${loaded}):`);
                this.write("  1. Train");
                this.write("  2. Save model");
                this.write("  3. Load model");
                this.write("  4. Load GPT-2 small weights");
                this.write("  5. Temperature");
                this.write("  6. Sampling method");
                this.write("  7. Chat");
                const choice = await this.ask(`Choose 1 to 7 (or "${GPTConsole.QUIT}" to exit): `);
                // Each option, and confirmQuit(), returns false if the input ended while it was asking for something, which also ends the program.
                if (choice === null) {
                    running = false;
                }
                else if (choice.toLowerCase() === GPTConsole.QUIT) {
                    running = !(await this.confirmQuit());
                }
                else if (choice === "1") {
                    running = await this.train();
                }
                else if (choice === "2") {
                    running = await this.saveModel();
                }
                else if (choice === "3") {
                    running = await this.loadModel();
                }
                else if (choice === "4") {
                    running = await this.loadGPT2();
                }
                else if (choice === "5") {
                    running = await this.setTemperature();
                }
                else if (choice === "6") {
                    running = await this.setSamplingMethod();
                }
                else if (choice === "7") {
                    running = await this.chat();
                }
                else {
                    this.write(`"${choice}" is not an option.`);
                }
            }
            this.write("Goodbye.");
        }
        finally {
            lineReader.close();
        }
    }
    /**
     * Checks that the user really wants to quit, if that would lose training:
     * the weights are kept in memory until they are saved, so training that
     * may have taken hours is lost when the program ends. If there is no
     * unsaved training, the program quits without asking.
     *
     * @returns True to quit: there is no unsaved training, the user answered "y" or "yes", or the input ended. False to stay in the program.
     */
    async confirmQuit() {
        if (!this.unsavedTraining) {
            return true;
        }
        this.write(`The model has been trained ${this.trainingRuns} time${this.trainingRuns === 1 ? "" : "s"}, ` +
            "and its training will be lost when the program ends. Choose 2 at the menu to save it.");
        const answer = await this.ask("Quit anyway? (y/n): ");
        if (answer === null) {
            return true;
        }
        return GPTConsole.isYes(answer);
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
    async train() {
        const hours = await this.askHours();
        if (hours === undefined) {
            return false;
        }
        if (hours === null) {
            this.write("Training cancelled.");
            return true;
        }
        const planned = this.hyperparametersForHours(hours);
        if (GPTConsole.sameArchitecture(planned, this.hyperparameters)) {
            // The current model is the right size: keep its weights, and train it with the new amount of data.
            GPTConsole.copyTrainingSettings(planned, this.hyperparameters);
        }
        else {
            // A model of a different size has to start again from random weights. Ask first if that would lose training or a loaded model.
            if (this.trainingRuns > 0 || this.loadedFrom !== null) {
                this.write(`Training for ${hours} hours needs a model ${planned.embeddingSize} values wide, but the current model is ` +
                    `${this.hyperparameters.embeddingSize} wide. A new model would start again from random weights, losing the current training.`);
                const answer = await this.ask("Start a new model? (y/n): ");
                if (answer === null) {
                    return false;
                }
                if (!GPTConsole.isYes(answer)) {
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
            this.unsavedTraining = true;
            const last = losses[losses.length - 1];
            this.write(`Training finished. Training loss: ${last.trainingLoss.toFixed(3)}. Validation loss: ${last.validationLoss.toFixed(3)}.`);
        }
        catch (error) {
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
    async askHours() {
        for (;;) {
            const answer = await this.ask(`How many hours of training? (e.g. 2.5; press Enter or type "${GPTConsole.QUIT}" to cancel): `);
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
            const hours = Number(answer);
            if (!Number.isFinite(hours)) {
                this.write(`"${answer}" is not a number of hours.`);
            }
            else if (hours <= 0) {
                this.write("The number of hours must be more than 0.");
            }
            else if (hours > GPTConsole.MAXIMUM_HOURS) {
                this.write(`The most the console accepts is ${GPTConsole.MAXIMUM_HOURS} hours.`);
            }
            else {
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
    writePlan(hours) {
        const finish = new Date(Date.now() + hours * 60 * 60 * 1000);
        const finishTime = finish.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
        const stories = this.hyperparameters.numberTrainingStories;
        const epochs = this.hyperparameters.numberEpochs;
        // Each run continues from where the trainer's last run finished in the training file.
        const firstStory = this.trainer.tinyStoriesLoaded + 1;
        this.write(`Training for about ${hours} hour${hours === 1 ? "" : "s"}: ${stories} stories ` +
            `(numbers ${firstStory} to ${firstStory + stories - 1}), ${epochs} epoch${epochs === 1 ? "" : "s"}, ` +
            `a model ${this.hyperparameters.embeddingSize} values wide with ` +
            `${(this.model.getParameterCount() / 1e6).toFixed(1)} million parameters.`);
        this.write(`Expected to finish at about ${finishTime}. The console is unavailable until then.`);
    }
    /**
     * Menu option 2: asks for a file name, checks it, and saves the model's
     * weights and hyperparameters there, showing the progress. If a file
     * already exists there the user is asked whether to overwrite it; if not,
     * or if the file name cannot be used, the user is asked for another. If
     * saving fails, the error is shown and the console returns to the menu.
     *
     * @returns True to return to the menu; false if the input ended, which ends the program.
     */
    async saveModel() {
        if (this.trainingRuns === 0 && this.loadedFrom === null) {
            this.write("(The model has not been trained yet, so the saved weights will be random.)");
        }
        let filePath;
        for (;;) {
            filePath = await this.askFilePath("save the model to", GPTConsole.checkSavePath);
            if (filePath === undefined) {
                return false;
            }
            if (filePath === null) {
                this.write("Save cancelled.");
                return true;
            }
            if (!fs.existsSync(filePath)) {
                break;
            }
            // The file exists: only overwrite it if the user says so; otherwise ask for another name.
            const answer = await this.ask(`A file already exists at ${filePath}. Overwrite it? (y/n): `);
            if (answer === null) {
                return false;
            }
            if (GPTConsole.isYes(answer)) {
                break;
            }
            this.write("The existing file has not been changed.");
        }
        const megabytes = ((this.model.getParameterCount() * 4) / 1e6).toFixed(1);
        this.write(`Saving the model (${(this.model.getParameterCount() / 1e6).toFixed(1)} million parameters, ${megabytes} MB) to ${filePath}`);
        const start = Date.now();
        try {
            this.withProgress("Saved", (onProgress) => this.modelManager.save(this.model, filePath, onProgress));
            this.unsavedTraining = false;
            this.write(`Model saved in ${((Date.now() - start) / 1000).toFixed(1)} seconds.`);
        }
        catch (error) {
            this.write(`Saving failed: ${error instanceof Error ? error.message : String(error)}`);
        }
        return true;
    }
    /**
     * Menu option 3: asks for a file name, checks that there is a file there,
     * and replaces the current model with the one saved in it (see
     * replaceModel). If the file name cannot be used, the user is asked for
     * another.
     *
     * @returns True to return to the menu; false if the input ended, which ends the program.
     */
    async loadModel() {
        const filePath = await this.askFilePath("load the model from", GPTConsole.checkLoadPath);
        if (filePath === undefined) {
            return false;
        }
        if (filePath === null) {
            this.write("Load cancelled.");
            return true;
        }
        return this.replaceModel(filePath, "the model", (onProgress) => this.modelManager.loadModel(filePath, Hyperparameters.GENERATION_MODE, onProgress));
    }
    /**
     * Menu option 4: replaces the current model with GPT-2 small, with
     * OpenAI's pretrained weights read from gpt2FilePath (see replaceModel),
     * and sets its replies to GPT2_OUTPUT_LENGTH tokens. If the weights have
     * not been downloaded, says where to get them and returns to the menu.
     *
     * @returns True to return to the menu; false if the input ended, which ends the program.
     */
    async loadGPT2() {
        if (!fs.existsSync(this.gpt2FilePath)) {
            this.write(`The GPT-2 small weights were not found at ${this.gpt2FilePath}. Download them with:`);
            this.write(`  curl.exe -L ${ModelManager.GPT2_SMALL_URL} -o "${this.gpt2FilePath}"`);
            return true;
        }
        return this.replaceModel(this.gpt2FilePath, "GPT-2 small's weights", (onProgress) => {
            const model = this.modelManager.loadGPT2(this.gpt2FilePath, Hyperparameters.GENERATION_MODE, onProgress);
            model.hyperparameters.outputLength = GPTConsole.GPT2_OUTPUT_LENGTH;
            return model;
        }, `Chat replies will be ${GPTConsole.GPT2_OUTPUT_LENGTH} tokens long and take about a minute each, as GPT-2 small is large for plain TypeScript.`);
    }
    /**
     * Replaces the current model with one loaded from a file, showing the
     * progress. Shared by Load model and Load GPT-2 small weights. If the
     * current model has training that has not been saved, the user is asked
     * first, since loading would lose it. If loading fails (for example, the
     * file is not a saved model), the error is shown, the current model is
     * kept, and the console returns to the menu.
     *
     * @param filePath - The file being loaded, shown to the user and in the menu afterwards.
     * @param description - What is being loaded, e.g. "the model", shown in the progress message.
     * @param load - Loads the model, calling the progress callback it is given.
     * @param note - Optional; shown after the model has loaded successfully.
     * @returns True to return to the menu; false if the input ended, which ends the program.
     */
    async replaceModel(filePath, description, load, note) {
        if (this.unsavedTraining) {
            this.write(`The current model has been trained ${this.trainingRuns} time${this.trainingRuns === 1 ? "" : "s"} since it was last saved, ` +
                "and its training will be lost if another model is loaded.");
            const answer = await this.ask("Load anyway? (y/n): ");
            if (answer === null) {
                return false;
            }
            if (!GPTConsole.isYes(answer)) {
                this.write("Load cancelled; the current model is unchanged.");
                return true;
            }
        }
        const megabytes = (fs.statSync(filePath).size / 1e6).toFixed(1);
        this.write(`Loading ${description} from ${filePath} (${megabytes} MB)`);
        const start = Date.now();
        try {
            const model = this.withProgress("Loaded", load);
            this.useModel(model);
            this.loadedFrom = filePath;
            this.write(`Model loaded in ${((Date.now() - start) / 1000).toFixed(1)} seconds: ` +
                `${(model.getParameterCount() / 1e6).toFixed(1)} million parameters, a model ${this.hyperparameters.embeddingSize} values wide ` +
                `with ${this.hyperparameters.numberTransformerBlocks} transformer blocks.`);
            if (note !== undefined) {
                this.write(note);
            }
        }
        catch (error) {
            this.write(`Loading failed: ${error instanceof Error ? error.message : String(error)}`);
            this.write("The current model is unchanged.");
        }
        return true;
    }
    /**
     * Asks for the path of a model file, repeating the question until the
     * check accepts the path. Relative paths are resolved from the folder the
     * program was started in, surrounding quotes (as added by Windows' "Copy
     * as path") are removed, and MODEL_EXTENSION is added if the name has no
     * extension and nothing exists at it. An empty answer or "quit" cancels.
     *
     * @param action - What the file is for, shown in the question, e.g. "save the model to".
     * @param check - Returns why a path cannot be used, or null if it can.
     * @returns The full path; null if the user cancelled; undefined if the input ended.
     */
    async askFilePath(action, check) {
        for (;;) {
            const answer = await this.ask(`File to ${action}? (e.g. models/tiny${GPTConsole.MODEL_EXTENSION}; press Enter or type "${GPTConsole.QUIT}" to cancel): `);
            if (answer === null) {
                return undefined;
            }
            if (answer === "" || answer.toLowerCase() === GPTConsole.QUIT) {
                return null;
            }
            // The extension is only added if nothing exists at the path as typed, so a folder's name is reported as a folder.
            let filePath = path.resolve(answer.replace(/^"(.*)"$/, "$1"));
            if (path.extname(filePath) === "" && !fs.existsSync(filePath)) {
                filePath += GPTConsole.MODEL_EXTENSION;
            }
            const problem = check(filePath);
            if (problem === null) {
                return filePath;
            }
            this.write(`${problem} Please enter a different file name.`);
        }
    }
    /**
     * Checks whether a model can be saved at a path: its folder must exist
     * and be writable, its name must be allowed, and it must not be a folder.
     * An existing file is acceptable; the caller asks before overwriting it.
     *
     * @param filePath - The full path to check.
     * @returns Why the path cannot be used, or null if it can.
     */
    static checkSavePath(filePath) {
        const folder = path.dirname(filePath);
        if (GPTConsole.INVALID_FILE_NAME_CHARACTERS.test(path.basename(filePath))) {
            return `"${path.basename(filePath)}" contains characters that are not allowed in file names.`;
        }
        if (!fs.existsSync(folder)) {
            return `The folder ${folder} does not exist.`;
        }
        if (!fs.statSync(folder).isDirectory()) {
            return `${folder} is a file, not a folder.`;
        }
        if (fs.existsSync(filePath) && fs.statSync(filePath).isDirectory()) {
            return `${filePath} is a folder, not a file.`;
        }
        try {
            fs.accessSync(folder, fs.constants.W_OK);
        }
        catch {
            return `The folder ${folder} cannot be written to.`;
        }
        return null;
    }
    /**
     * Checks whether a model can be loaded from a path: there must be a
     * readable file there. Whether the file really is a saved model is only
     * found out when it is loaded.
     *
     * @param filePath - The full path to check.
     * @returns Why the path cannot be used, or null if it can.
     */
    static checkLoadPath(filePath) {
        if (!fs.existsSync(filePath)) {
            return `There is no file at ${filePath}.`;
        }
        if (fs.statSync(filePath).isDirectory()) {
            return `${filePath} is a folder, not a file.`;
        }
        try {
            fs.accessSync(filePath, fs.constants.R_OK);
        }
        catch {
            return `${filePath} cannot be read.`;
        }
        return null;
    }
    /**
     * Runs a save or load, showing its progress on one line in steps of 10%,
     * e.g. "Saved: 10% 20% ... 100%". The line is always finished, even if the
     * save or load fails part way.
     *
     * @param label - The word shown at the start of the line, e.g. "Saved".
     * @param action - Does the saving or loading, calling the progress callback it is given.
     * @returns Whatever action returns.
     */
    withProgress(label, action) {
        this.output.write(`${label}:`);
        let tenthsShown = 0;
        try {
            return action((bytesDone, bytesTotal) => {
                // Large arrays (e.g. the token embedding) can cover several tenths at once, so each one passed is shown.
                const tenths = bytesTotal === 0 ? 10 : Math.floor((bytesDone / bytesTotal) * 10);
                while (tenthsShown < tenths) {
                    tenthsShown++;
                    this.output.write(` ${tenthsShown * 10}%`);
                }
            });
        }
        finally {
            this.write("");
        }
    }
    /**
     * Menu option 5: shows the current temperature and asks for a new one,
     * repeating the question until the answer is a number between 0 and 1. An
     * empty answer or "quit" leaves the temperature unchanged.
     *
     * @returns True to return to the menu; false if the input ended, which ends the program.
     */
    async setTemperature() {
        this.write(`The temperature is currently ${this.hyperparameters.temperature}.`);
        // Greedy sampling always takes the most likely token, so the temperature only matters for top-k and top-p.
        if (this.hyperparameters.samplingMethod === Hyperparameters.GREEDY_SAMPLING) {
            this.write("(Greedy sampling ignores the temperature; choose 6 at the menu to use top-k or top-p.)");
        }
        for (;;) {
            const answer = await this.ask(`New temperature? (0 to 1; press Enter or type "${GPTConsole.QUIT}" to cancel): `);
            if (answer === null) {
                return false;
            }
            if (answer === "" || answer.toLowerCase() === GPTConsole.QUIT) {
                this.write("The temperature is unchanged.");
                return true;
            }
            // Number() rejects anything that is not entirely a number (see askHours).
            const temperature = Number(answer);
            if (temperature >= 0 && temperature <= 1) {
                this.hyperparameters.temperature = temperature;
                this.write(`The temperature is now ${temperature}.`);
                return true;
            }
            this.write(`"${answer}" is not a number between 0 and 1. Please enter a temperature from 0 to 1, or "${GPTConsole.QUIT}".`);
        }
    }
    /**
     * Menu option 6: shows the current sampling method and asks for a new one,
     * typed as the method's name followed by its setting, if it has one:
     * "greedy", "top-p 0.9" or "top-k 30". If the setting is left out, the
     * current value is kept. The question is repeated until the answer can be
     * used. An empty answer or "quit" leaves the method unchanged.
     *
     * @returns True to return to the menu; false if the input ended, which ends the program.
     */
    async setSamplingMethod() {
        this.write(`The sampling method is currently ${this.describeSamplingMethod()}. The methods are:`);
        this.write("  greedy       always choose the most likely next token");
        this.write("  top-p <p>    choose at random from the most likely tokens whose probabilities add up to p (above 0, up to 1)");
        this.write("  top-k <k>    choose at random from the k most likely tokens (a whole number, 1 or more)");
        for (;;) {
            const answer = await this.ask(`Sampling method? (e.g. "top-k 30"; press Enter or type "${GPTConsole.QUIT}" to cancel): `);
            if (answer === null) {
                return false;
            }
            if (answer === "" || answer.toLowerCase() === GPTConsole.QUIT) {
                this.write("The sampling method is unchanged.");
                return true;
            }
            const problem = this.applySamplingMethod(answer);
            if (problem === null) {
                this.write(`The sampling method is now ${this.describeSamplingMethod()}.`);
                return true;
            }
            this.write(`${problem} Please enter "greedy", "top-p <p>" or "top-k <k>", or "${GPTConsole.QUIT}".`);
        }
    }
    /**
     * Reads a sampling method typed by the user, e.g. "top-k 30", and if it is
     * valid sets the hyperparameters to it. Nothing is changed if it is not.
     *
     * @param answer - The text typed: a method name, optionally followed by its setting.
     * @returns Why the answer cannot be used, or null if the method has been set.
     */
    applySamplingMethod(answer) {
        const [name, setting, ...extra] = answer.toLowerCase().split(/\s+/);
        if (extra.length > 0) {
            return `"${answer}" has too many parts.`;
        }
        if (name === "greedy") {
            if (setting !== undefined) {
                return "Greedy sampling does not take a setting.";
            }
            this.hyperparameters.samplingMethod = Hyperparameters.GREEDY_SAMPLING;
            return null;
        }
        if (name === "top-k") {
            // Number() rejects anything that is not entirely a number (see askHours).
            const k = setting === undefined ? this.hyperparameters.topK : Number(setting);
            if (!(Number.isInteger(k) && k > 0)) {
                return `"${setting}" is not a whole number of tokens, 1 or more.`;
            }
            this.hyperparameters.topK = k;
            this.hyperparameters.samplingMethod = Hyperparameters.TOP_K_SAMPLING;
            return null;
        }
        if (name === "top-p") {
            const p = setting === undefined ? this.hyperparameters.topP : Number(setting);
            // Written as a negated range check so that NaN, from text that is not a number, is rejected.
            if (!(p > 0 && p <= 1)) {
                return `"${setting}" is not a number above 0 and up to 1.`;
            }
            this.hyperparameters.topP = p;
            this.hyperparameters.samplingMethod = Hyperparameters.TOP_P_SAMPLING;
            return null;
        }
        return `"${name}" is not a sampling method.`;
    }
    /**
     * Describes the current sampling method with its setting, for messages.
     *
     * @returns e.g. "greedy", "top-k (k = 40)" or "top-p (p = 0.9)".
     */
    describeSamplingMethod() {
        switch (this.hyperparameters.samplingMethod) {
            case Hyperparameters.TOP_K_SAMPLING:
                return `top-k (k = ${this.hyperparameters.topK})`;
            case Hyperparameters.TOP_P_SAMPLING:
                return `top-p (p = ${this.hyperparameters.topP})`;
            default:
                return "greedy";
        }
    }
    /**
     * Menu option 7: reads sentences from the user and shows the model's
     * continuation of each, until the user types "quit". Empty lines are
     * ignored.
     *
     * @returns True to return to the menu; false if the input ended, which ends the program.
     */
    async chat() {
        this.write(`Chat: type the start of a sentence and the model will continue it. Type "${GPTConsole.QUIT}" to return to the menu.`);
        if (this.trainingRuns === 0 && this.loadedFrom === null) {
            this.write("(The model has not been trained yet, so its replies will be random.)");
        }
        for (;;) {
            const sentence = await this.ask("You: ");
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
    reply(sentence) {
        const continuation = this.generator.generate(sentence).slice(sentence.length);
        const endOfStory = continuation.indexOf(BPETokenizer.END_OF_TEXT);
        return endOfStory === -1 ? continuation : continuation.slice(0, endOfStory) + " [end of story]";
    }
    /**
     * Builds a new, untrained model from the given hyperparameters, with its
     * own trainer and generator, replacing the current ones.
     *
     * @param hyperparameters - The settings for the new model.
     */
    buildModel(hyperparameters) {
        // The generator needs generation mode; the trainer switches to training mode while it trains and then switches back.
        hyperparameters.mode = Hyperparameters.GENERATION_MODE;
        this.useModel(new GPTModel(hyperparameters));
    }
    /**
     * Makes a model the current one, with a new trainer and generator built
     * from its hyperparameters, and resets the record of its training. Used
     * for both newly built and loaded models; the caller sets loadedFrom for a
     * loaded one. The temperature and sampling method chosen in the console
     * are carried over from the previous model, so replacing the model does
     * not quietly undo them.
     *
     * @param model - The model, in generation mode.
     */
    useModel(model) {
        // The constructor's call has no previous model to carry settings over from.
        if (this.hyperparameters !== undefined) {
            GPTConsole.copyDecodingSettings(this.hyperparameters, model.hyperparameters);
        }
        this.model = model;
        this.hyperparameters = model.hyperparameters;
        this.trainer = new Trainer(this.hyperparameters, this.model);
        this.generator = new Generator(this.hyperparameters, this.model);
        this.trainingRuns = 0;
        this.unsavedTraining = false;
        this.loadedFrom = null;
    }
    /**
     * Checks whether the user answered yes to a y/n question.
     *
     * @param answer - The answer typed.
     * @returns True for "y" or "yes", in any case.
     */
    static isYes(answer) {
        return answer.toLowerCase() === "y" || answer.toLowerCase() === "yes";
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
    static sameArchitecture(a, b) {
        return (a.embeddingSize === b.embeddingSize &&
            a.numberAttentionHeads === b.numberAttentionHeads &&
            a.numberTransformerBlocks === b.numberTransformerBlocks &&
            a.contextLength === b.contextLength &&
            a.qkvBias === b.qkvBias &&
            a.feedForwardBias === b.feedForwardBias);
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
    static copyTrainingSettings(from, to) {
        to.numberTrainingStories = from.numberTrainingStories;
        to.numberValidationStories = from.numberValidationStories;
        to.numberEpochs = from.numberEpochs;
        to.batchSize = from.batchSize;
        to.learningRate = from.learningRate;
        to.outputLength = from.outputLength;
    }
    /**
     * Copies the settings that control how the next token is chosen when
     * generating text from one set of hyperparameters to another.
     *
     * @param from - The hyperparameters of the model being replaced.
     * @param to - The hyperparameters of the new model.
     */
    static copyDecodingSettings(from, to) {
        to.temperature = from.temperature;
        to.samplingMethod = from.samplingMethod;
        to.topK = from.topK;
        to.topP = from.topP;
    }
    /**
     * Shows a prompt and waits for the user to type a line.
     *
     * @param prompt - The text shown before the user's typing, on the same line.
     * @returns The line typed, with surrounding spaces removed, or null if the input has ended.
     */
    async ask(prompt) {
        this.output.write(prompt);
        const result = await this.lines.next();
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
    write(text) {
        this.output.write(text + "\n");
    }
}
export default GPTConsole;
