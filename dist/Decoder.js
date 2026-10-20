import Hyperparameters from "./Hyperparameters.js";
/**
 * Chooses the next token from the model's output, an array of probabilities
 * with one entry per token in the vocabulary, and returns the chosen token's
 * id. Each method implements a different strategy for choosing the token.
 * Converting ids back into text is left to the caller (e.g. Generator), so
 * that the generated ids can be decoded together once generation is finished.
 */
class Decoder {
    /**
     * The Hyperparameters this decoder was built with. A reference is kept,
     * rather than copies of its values, so that switching to generation mode
     * after construction is seen by the decoder.
     */
    hyperparameters;
    /** Draws from the Hyperparameters' shared random number stream, used to break ties between tokens. */
    random;
    /**
     * Creates a Decoder. Ties are broken using the Hyperparameters' shared
     * random number stream, so the choices are reproducible when
     * Hyperparameters.seed is set and differ on every run when it is not.
     *
     * @param hyperparameters - Source of the mode, vocabularySize and the random number stream. Defaults to a new Hyperparameters instance.
     */
    constructor(hyperparameters = new Hyperparameters()) {
        this.hyperparameters = hyperparameters;
        this.random = () => hyperparameters.random();
    }
    /**
     * Greedy decoding: picks the token with the highest probability and
     * returns its id. If several tokens share the highest probability, one
     * of them is chosen at random.
     *
     * @param probabilities - One probability per token in the vocabulary, indexed by token id. Must be vocabularySize long.
     * @returns The id of the chosen token.
     * @throws Error if the hyperparameters are not in generation mode, since text is only generated in that mode.
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
        for (let tokenId = 0; tokenId < probabilities.length; tokenId++) {
            const probability = probabilities[tokenId];
            if (probability > highestProbability) {
                highestProbability = probability;
                bestTokenIds = [tokenId];
            }
            else if (probability === highestProbability) {
                bestTokenIds.push(tokenId);
            }
        }
        // bestTokenIds is empty only if every value is NaN, which fails every comparison.
        if (bestTokenIds.length === 0) {
            throw new RangeError("Decoder| probabilities contains no valid values");
        }
        return bestTokenIds[Math.floor(this.random() * bestTokenIds.length)];
    }
    /**
     * Checks that decoding is allowed and that the probabilities cover the
     * whole vocabulary.
     *
     * @param probabilities - The probabilities to check.
     * @throws Error if the hyperparameters are not in generation mode.
     * @throws RangeError if probabilities is not vocabularySize long.
     */
    validate(probabilities) {
        if (this.hyperparameters.mode !== Hyperparameters.GENERATION_MODE) {
            throw new Error("Decoder| Mode must be GENERATION_MODE to decode tokens");
        }
        if (probabilities.length !== this.hyperparameters.vocabularySize) {
            throw new RangeError(`Decoder| Expected ${this.hyperparameters.vocabularySize} probabilities; got ${probabilities.length}`);
        }
    }
}
export default Decoder;
