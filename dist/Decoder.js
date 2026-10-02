import BPETokenizer from "./BPETokenizer.js";
import Hyperparameters from "./Hyperparameters.js";
/**
 * Converts the model's output, an array of probabilities with one entry per
 * token in the vocabulary, into text. Each method implements a different
 * strategy for choosing which token to output.
 */
class Decoder {
    /**
     * The Hyperparameters this decoder was built with. A reference is kept,
     * rather than copies of its values, so that switching training mode off
     * after construction is seen by the decoder.
     */
    hyperparameters;
    /** Converts the chosen token id back into text. */
    tokenizer = new BPETokenizer();
    /** Draws from the Hyperparameters' shared random number stream, used to break ties between tokens. */
    random;
    /**
     * Creates a Decoder. Ties are broken using the Hyperparameters' shared
     * random number stream, so the choices are reproducible when
     * Hyperparameters.seed is set and differ on every run when it is not.
     *
     * @param hyperparameters - Source of the training flag, vocabularySize and the random number stream. Defaults to a new Hyperparameters instance.
     */
    constructor(hyperparameters = new Hyperparameters()) {
        this.hyperparameters = hyperparameters;
        this.random = () => hyperparameters.random();
    }
    /**
     * Greedy decoding: picks the token with the highest probability and
     * returns its text. If several tokens share the highest probability, one
     * of them is chosen at random.
     *
     * @param probabilities - One probability per token in the vocabulary, indexed by token id. Must be vocabularySize long.
     * @returns The text of the chosen token.
     * @throws Error if the hyperparameters are in training mode, since text is only generated during inference.
     * @throws RangeError if probabilities is not vocabularySize long.
     */
    greedy(probabilities) {
        this.validate(probabilities);
        /*
         * Single pass that tracks the highest probability seen so far and every
         * token id that has it. A new maximum resets the list; an equal value
         * is added to it, so ties are collected without a second scan.
         */
        let highestProbability = -Infinity;
        let bestTokenIds = [];
        probabilities.forEach((probability, tokenId) => {
            if (probability > highestProbability) {
                highestProbability = probability;
                bestTokenIds = [tokenId];
            }
            else if (probability === highestProbability) {
                bestTokenIds.push(tokenId);
            }
        });
        // bestTokenIds is empty only if every value is NaN, which fails every comparison.
        if (bestTokenIds.length === 0) {
            throw new RangeError("Decoder| probabilities contains no valid values");
        }
        const tokenId = bestTokenIds[Math.floor(this.random() * bestTokenIds.length)];
        return this.tokenizer.decode([tokenId]);
    }
    /**
     * Checks that decoding is allowed and that the probabilities cover the
     * whole vocabulary.
     *
     * @param probabilities - The probabilities to check.
     * @throws Error if the hyperparameters are in training mode.
     * @throws RangeError if probabilities is not vocabularySize long.
     */
    validate(probabilities) {
        if (this.hyperparameters.training) {
            throw new Error("Decoder| Training mode must be off to decode tokens");
        }
        if (probabilities.length !== this.hyperparameters.vocabularySize) {
            throw new RangeError(`Decoder| Expected ${this.hyperparameters.vocabularySize} probabilities; got ${probabilities.length}`);
        }
    }
}
export default Decoder;
