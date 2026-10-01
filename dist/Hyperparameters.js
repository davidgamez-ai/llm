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
    _bias;
    _positionEmbeddingStandardDeviation;
    /**
     * Creates a Hyperparameters instance.
     *
     * @param embeddingSize - The size of each token embedding vector.
     * @param contextLength - The number of tokens of context the model
     * operates over at once.
     * @param dropoutRate - The probability, between 0 and 1 inclusive, that
     * any given value is dropped (set to zero) during dropout.
     * @param training - True for training mode, in which dropout is applied;
     * false for inference mode, in which dropout is skipped.
     * @param numberAttentionHeads - The number of attention heads used in
     * multi-head attention. Must be a positive whole number that divides
     * embeddingSize exactly.
     * @param numberTransformerBlocks - The number of transformer blocks stacked
     * in the model. Must be a positive whole number.
     * @param batchSize - The number of input sequences processed together in
     * one batch. Must be a positive whole number.
     * @param bias - True if the linear layers of the feed forward network add
     * a bias to their outputs; false if they do not.
     * @param positionEmbeddingStandardDeviation - The standard deviation of the
     * normal distribution (mean 0) used to initialize the position embedding
     * matrix. Defaults to 0.01, the value used by the original GPT-2. Must be
     * a finite number greater than or equal to 0.
     * @throws RangeError if dropoutRate is less than 0, greater than 1 or NaN.
     * @throws RangeError if numberAttentionHeads is not a positive whole number.
     * @throws RangeError if embeddingSize divided by numberAttentionHeads is
     * not a whole number.
     * @throws RangeError if numberTransformerBlocks is not a positive whole number.
     * @throws RangeError if batchSize is not a positive whole number.
     * @throws RangeError if positionEmbeddingStandardDeviation is negative,
     * infinite or NaN.
     */
    constructor(embeddingSize = 768, contextLength = 1024, dropoutRate = 0.1, training = true, numberAttentionHeads = 12, numberTransformerBlocks = 12, batchSize = 2, bias = false, positionEmbeddingStandardDeviation = 0.01) {
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
        /*
         * Each head produces context vectors weightMatrixColumns wide, and
         * MultiHeadAttention concatenates the heads' outputs, so the combined
         * width only equals embeddingSize if the heads split it evenly.
         */
        const weightMatrixColumns = embeddingSize / numberAttentionHeads;
        if (!Number.isInteger(weightMatrixColumns)) {
            throw new RangeError(`Hyperparameters| embeddingSize (${embeddingSize}) divided by numberAttentionHeads ` +
                `(${numberAttentionHeads}) must be a whole number; got ${weightMatrixColumns}`);
        }
        // Same positive whole number check as numberAttentionHeads.
        if (!(Number.isInteger(numberTransformerBlocks) && numberTransformerBlocks > 0)) {
            throw new RangeError(`Hyperparameters| numberTransformerBlocks must be a positive whole number; got ${numberTransformerBlocks}`);
        }
        // Same positive whole number check as numberAttentionHeads.
        if (!(Number.isInteger(batchSize) && batchSize > 0)) {
            throw new RangeError(`Hyperparameters| batchSize must be a positive whole number; got ${batchSize}`);
        }
        // Number.isFinite rejects NaN and Infinity; a negative standard deviation is meaningless.
        if (!(Number.isFinite(positionEmbeddingStandardDeviation) && positionEmbeddingStandardDeviation >= 0)) {
            throw new RangeError(`Hyperparameters| positionEmbeddingStandardDeviation must be a finite number >= 0; ` +
                `got ${positionEmbeddingStandardDeviation}`);
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
        this._bias = bias;
        this._positionEmbeddingStandardDeviation = positionEmbeddingStandardDeviation;
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
     * i.e. the size of the projected vectors those matrices produce. Equal to
     * embeddingSize / numberAttentionHeads, so the concatenated output of all
     * the attention heads is embeddingSize wide.
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
     * True if the linear layers of the feed forward network add a bias to
     * their outputs; false if they do not.
     */
    get bias() {
        return this._bias;
    }
    /**
     * The standard deviation of the normal distribution (mean 0) used to
     * initialize the position embedding matrix. The original GPT-2 uses 0.01.
     */
    get positionEmbeddingStandardDeviation() {
        return this._positionEmbeddingStandardDeviation;
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
