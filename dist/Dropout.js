import Hyperparameters from "./Hyperparameters.js";
/**
 * Dropout: during training, sets each value to zero with probability
 * dropoutRate and scales the values that are kept by 1 / (1 - dropoutRate),
 * so the expected size of each value is unchanged. Randomly removing values
 * stops the model relying too heavily on any one of them, which reduces
 * overfitting. In validation and generation modes dropout does nothing.
 *
 * Dropout is random, so the backward pass must use exactly the same mask as
 * the forward pass: a value that was dropped had no effect on the output, so
 * it must receive no error. Each place in the model that applies dropout
 * therefore has its own Dropout instance, which keeps the mask from its most
 * recent update() for backpropagate().
 */
class Dropout {
    /** Probability, between 0 and 1, that each value is dropped (set to zero). */
    dropoutRate;
    /**
     * The current mode: Hyperparameters.TRAINING_MODE, VALIDATION_MODE or
     * GENERATION_MODE. Dropout is only applied in TRAINING_MODE. Read from
     * Hyperparameters on every access, so changing Hyperparameters.mode after
     * construction takes effect.
     */
    get mode() {
        return this.getMode();
    }
    /**
     * The mask used by the most recent update(): one row per row of values,
     * each entry either 0 (dropped) or 1 / (1 - dropoutRate) (kept). Null when
     * the most recent update() was not in training mode, or before the first
     * update().
     */
    get mask() {
        return this._mask;
    }
    /** Backing field for mask. */
    _mask = null;
    /** Reads the Hyperparameters' current mode. */
    getMode;
    /** Draws from the Hyperparameters' shared random number stream, which is seeded when Hyperparameters.seed is set. */
    random;
    /**
     * Creates a Dropout, copying dropoutRate from the given Hyperparameters.
     *
     * @param hyperparameters - Source of dropoutRate, mode and the random number stream. Defaults to a new Hyperparameters instance.
     */
    constructor(hyperparameters = new Hyperparameters()) {
        this.dropoutRate = hyperparameters.dropoutRate;
        this.getMode = () => hyperparameters.mode;
        this.random = () => hyperparameters.random();
    }
    /**
     * Applies dropout, only in training mode. A mask with the same shape as
     * values is built, row by row, with one random number drawn per value:
     * each entry is 0 with probability dropoutRate, and 1 / (1 - dropoutRate)
     * otherwise. The values are then multiplied by the mask, element by
     * element, and the mask is stored for backpropagate(). With the default
     * dropoutRate of 0.1, kept values are multiplied by 1 / 0.9, about 1.11.
     *
     * In validation and generation modes no mask is built, the stored mask is
     * cleared and the values are returned unchanged.
     *
     * @param values - The values to apply dropout to, e.g. one row per token.
     * @returns A new matrix with dropout applied in training mode; otherwise values itself.
     */
    update(values) {
        if (this.mode !== Hyperparameters.TRAINING_MODE) {
            this._mask = null;
            return values;
        }
        const keepScale = 1 / (1 - this.dropoutRate);
        this._mask = values.map((row) => row.map(() => (this.random() < this.dropoutRate ? 0 : keepScale)));
        const mask = this._mask;
        return values.map((row, i) => row.map((value, j) => value * mask[i][j]));
    }
    /**
     * Passes errors back through dropout. Each output value was its input
     * value times the mask entry, so the error at each input is the error at
     * the output times the same mask entry: dropped values receive no error,
     * and the errors of kept values are scaled by 1 / (1 - dropoutRate), just
     * as the values were.
     *
     * If the most recent update() was not in training mode, dropout did
     * nothing, so the errors are returned unchanged.
     *
     * @param outputErrors - The errors at the output of dropout: the same shape as the values given to the most recent update().
     * @returns The errors at the input of dropout, the same shape as outputErrors.
     * @throws RangeError if outputErrors is not the same shape as the stored mask.
     */
    backpropagate(outputErrors) {
        const mask = this._mask;
        if (mask === null) {
            return outputErrors;
        }
        if (outputErrors.length !== mask.length || outputErrors.some((row, i) => row.length !== mask[i].length)) {
            throw new RangeError("Dropout| The errors must have the same shape as the values given to the most recent update()");
        }
        return outputErrors.map((row, i) => row.map((error, j) => error * mask[i][j]));
    }
}
export default Dropout;
