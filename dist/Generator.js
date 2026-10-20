import BPETokenizer from "./BPETokenizer.js";
import Decoder from "./Decoder.js";
/**
 * Generates text with a GPTModel. The prompt is encoded into token ids once.
 * Then, until outputLength tokens have been generated, the ids are fed into
 * the model, the logits it returns are converted into probabilities, the
 * decoder chooses the next token's id from them, and that id is appended to
 * the ids. Finally the generated ids are converted back into text.
 *
 * Working with ids throughout, as the book does, avoids converting the text
 * back and forth on every step. That would not be safe: byte pair encoding
 * does not always turn decoded text back into the same ids, because the new
 * token's text can merge with the text before it.
 *
 * Tokens are currently chosen with greedy decoding.
 */
class Generator {
    /**
     * The Hyperparameters this generator was built with. A reference is kept,
     * rather than copies of its values, so that changes to outputLength after
     * construction are seen by generate().
     */
    hyperparameters;
    /** Converts the prompt into token ids, and the generated ids back into text. */
    tokenizer;
    /** The model that produces a score (logit) for every possible next token. */
    model;
    /** Chooses the id of the next token from the probabilities. */
    decoder;
    /**
     * Creates a Generator for the given model, building its own tokenizer and
     * decoder. The model is passed in rather than built here so that the same
     * instance can be shared with a Trainer, and text is generated with the
     * trained weights.
     *
     * hyperparameters.mode must be Hyperparameters.GENERATION_MODE when
     * generate() is called; otherwise the decoder throws.
     *
     * @param hyperparameters - The settings the model was built with; also the source of outputLength, contextLength and the settings used to build the decoder.
     * @param model - The model used to generate text. Must have been built from the same hyperparameters.
     */
    constructor(hyperparameters, model) {
        this.hyperparameters = hyperparameters;
        this.tokenizer = new BPETokenizer();
        this.model = model;
        this.decoder = new Decoder(hyperparameters);
    }
    /**
     * Generates hyperparameters.outputLength tokens following the given text.
     *
     * @param text - The prompt to continue.
     * @returns The prompt followed by the generated text.
     * @throws Error if the prompt encodes to no tokens, since the model then has nothing to predict from.
     * @throws Error if the hyperparameters are not in generation mode (thrown by Decoder).
     */
    generate(text) {
        // The prompt is encoded once; after that only token ids are used.
        const tokenIds = this.tokenizer.encode(text);
        if (tokenIds.length === 0) {
            throw new Error("Generator| The prompt must contain at least one token");
        }
        // The ids of the generated tokens, kept separately so they can be decoded at the end.
        const generatedIds = [];
        for (let t = 0; t < this.hyperparameters.outputLength; ++t) {
            /*
             * Only the last contextLength tokens are fed to the model, since the
             * position embedding has no rows beyond the context window. Once the
             * sequence grows longer than that, the earliest tokens are dropped.
             */
            const context = [...tokenIds, ...generatedIds].slice(-this.hyperparameters.contextLength);
            const logits = this.model.update(context);
            /*
             * Only the last row of logits is needed: it scores every candidate for
             * the token that follows the whole input. In generation mode the model
             * only returns that row, but taking the last row would also work if it
             * returned logits for every token.
             */
            const probabilities = Generator.softmax(logits[logits.length - 1]);
            generatedIds.push(this.decoder.greedy(probabilities));
        }
        /*
         * Decode all the generated ids together rather than one at a time. A
         * single character can be split across several tokens (e.g. some
         * non-English characters and emoji), and decoding those tokens
         * separately would garble it. The original prompt text is kept as it
         * was given.
         */
        return text + this.tokenizer.decode(generatedIds);
    }
    /**
     * Converts a row of logits into a probability distribution over the
     * vocabulary: every value positive and the row summing to 1.
     *
     * The row's maximum is subtracted before exponentiating to avoid
     * overflow; this does not change the result because softmax is unaffected
     * by adding a constant to every input. The maximum is found with reduce
     * rather than Math.max(...logits), because spreading a vocabulary-sized row
     * as arguments can exceed the engine's argument limit.
     *
     * @param logits - One logit per token in the vocabulary, indexed by token id.
     * @returns One probability per token in the vocabulary, indexed by token id.
     */
    static softmax(logits) {
        const maxLogit = logits.reduce((max, value) => Math.max(max, value), -Infinity);
        const exponentiated = logits.map((value) => Math.exp(value - maxLogit));
        const total = exponentiated.reduce((sum, value) => sum + value, 0);
        return exponentiated.map((value) => value / total);
    }
}
export default Generator;
