import BPETokenizer from "./BPETokenizer.js";
/**
 * Central store for hyperparameters shared across multiple parts of the
 * application (e.g. Embedding), so that a single instance can be passed
 * around instead of each consumer redeclaring its own copies of the same
 * values.
 */
class Hyperparameters {
    _embeddingSize;
    _contextLength;
    _vocabularySize;
    _weightMatrixColumns;
    _dropoutRate;
    _training;
    _numberAttentionHeads;
    _numberTransformerBlocks;
    _batchSize;
    /**
     * Creates a Hyperparameters instance.
     *
     * @param embeddingSize - The size of each token embedding vector.
     * @param contextLength - The number of tokens of context the model
     * operates over at once.
     * @param weightMatrixColumns - The number of columns in each of the
     * query/key/value weight matrices, i.e. the size of the projected
     * vectors those matrices produce.
     * @param dropoutRate - The probability, between 0 and 1 inclusive, that
     * any given value is dropped (set to zero) during dropout.
     * @param training - True for training mode, in which dropout is applied;
     * false for inference mode, in which dropout is skipped.
     * @param numberAttentionHeads - The number of attention heads used in
     * multi-head attention. Must be a positive whole number.
     * @param numberTransformerBlocks - The number of transformer blocks stacked
     * in the model. Must be a positive whole number.
     * @param batchSize - The number of input sequences processed together in
     * one batch. Must be a positive whole number.
     * @throws RangeError if dropoutRate is less than 0, greater than 1 or NaN.
     * @throws RangeError if numberAttentionHeads is not a positive whole number.
     * @throws RangeError if numberTransformerBlocks is not a positive whole number.
     * @throws RangeError if batchSize is not a positive whole number.
     */
    constructor(embeddingSize = 3, contextLength = 1024, weightMatrixColumns = 2, dropoutRate = 0.0, training = true, numberAttentionHeads = 5, numberTransformerBlocks = 12, batchSize = 2) {
        // Written as a negated range check so that NaN, which fails every
        // comparison, is rejected along with values outside [0, 1].
        if (!(dropoutRate >= 0 && dropoutRate <= 1)) {
            throw new RangeError(`Hyperparameters| dropoutRate must be between 0 and 1; got ${dropoutRate}`);
        }
        // Number.isInteger rejects fractions, NaN and Infinity, so combined
        // with the > 0 check only positive whole numbers are accepted.
        if (!(Number.isInteger(numberAttentionHeads) && numberAttentionHeads > 0)) {
            throw new RangeError(`Hyperparameters| numberAttentionHeads must be a positive whole number; got ${numberAttentionHeads}`);
        }
        // Same positive whole number check as numberAttentionHeads.
        if (!(Number.isInteger(numberTransformerBlocks) && numberTransformerBlocks > 0)) {
            throw new RangeError(`Hyperparameters| numberTransformerBlocks must be a positive whole number; got ${numberTransformerBlocks}`);
        }
        // Same positive whole number check as numberAttentionHeads.
        if (!(Number.isInteger(batchSize) && batchSize > 0)) {
            throw new RangeError(`Hyperparameters| batchSize must be a positive whole number; got ${batchSize}`);
        }
        this._embeddingSize = embeddingSize;
        this._contextLength = contextLength;
        this._vocabularySize = new BPETokenizer().vocabularySize;
        this._weightMatrixColumns = weightMatrixColumns;
        this._dropoutRate = dropoutRate;
        this._training = training;
        this._numberAttentionHeads = numberAttentionHeads;
        this._numberTransformerBlocks = numberTransformerBlocks;
        this._batchSize = batchSize;
    }
    /** The size of each token embedding vector. */
    get embeddingSize() {
        return this._embeddingSize;
    }
    /** The number of tokens of context the model operates over at once. */
    get contextLength() {
        return this._contextLength;
    }
    /**
     * The number of columns in each of the query/key/value weight matrices,
     * i.e. the size of the projected vectors those matrices produce.
     */
    get weightMatrixColumns() {
        return this._weightMatrixColumns;
    }
    /**
     * The probability, between 0 and 1 inclusive, that any given value is
     * dropped (set to zero) during dropout.
     */
    get dropoutRate() {
        return this._dropoutRate;
    }
    /**
     * True for training mode, in which dropout is applied; false for
     * inference mode, in which dropout is skipped.
     */
    get training() {
        return this._training;
    }
    /** The number of attention heads used in multi-head attention. */
    get numberAttentionHeads() {
        return this._numberAttentionHeads;
    }
    /** The number of transformer blocks stacked in the model. */
    get numberTransformerBlocks() {
        return this._numberTransformerBlocks;
    }
    /** The number of input sequences processed together in one batch. */
    get batchSize() {
        return this._batchSize;
    }
    /**
     * The size of the vocabulary, loaded from BPETokenizer in the
     * constructor so it always matches the tokenizer actually in use.
     */
    get vocabularySize() {
        return this._vocabularySize;
    }
}
export default Hyperparameters;
