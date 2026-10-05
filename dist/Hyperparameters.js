import BPETokenizer from "./BPETokenizer.js";
import createSeededRandom from "./SeededRandom.js";
/**
 * Central store for hyperparameters shared across multiple parts of the
 * application (e.g. Embedding), so that a single instance can be passed
 * around instead of each consumer redeclaring its own copies of the same
 * values.
 *
 * Values can be changed after construction through the setters, which
 * apply the same validation as the constructor. Changing a value does not
 * affect components that have already been built from this instance (e.g.
 * weight matrices are not resized), so setters should normally be used
 * before those components are created.
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
    _qkvBias;
    _feedForwardBias;
    _positionEmbeddingStandardDeviation;
    _tokenEmbeddingStandardDeviation;
    _attentionProjectionStandardDeviation;
    _outputLayerStandardDeviation;
    _outputLength;
    _seed;
    /**
     * The random number stream shared by every component built from this
     * instance: seeded from _seed when it is set, otherwise Math.random.
     * Replaced whenever the seed is set, so the stream restarts from the
     * beginning.
     */
    _random;
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
     * @param qkvBias - True if the query, key and value projections in each
     * attention head add a bias to their outputs; false if they do not.
     * Defaults to false, as in the book's GPT-2 configuration.
     * @param feedForwardBias - True if the linear layers of the feed forward
     * network add a bias to their outputs; false if they do not. Defaults to
     * true, as in GPT-2.
     * @param positionEmbeddingStandardDeviation - The standard deviation of the
     * normal distribution (mean 0) used to initialize the position embedding
     * matrix. Defaults to 0.01, the value used by the original GPT-2. Must be
     * a finite number greater than or equal to 0.
     * @param tokenEmbeddingStandardDeviation - The standard deviation of the
     * normal distribution (mean 0) used to initialize the token embedding
     * matrix. Defaults to 0.02, as in GPT-2. Must be a finite number greater
     * than or equal to 0.
     * @param attentionProjectionStandardDeviation - The base standard deviation
     * of the normal distribution (mean 0) used to initialize the multi-head
     * attention output projection, before it is divided by
     * sqrt(2 * numberTransformerBlocks). Defaults to 0.02, as in GPT-2. Must be
     * a finite number greater than or equal to 0.
     * @param outputLayerStandardDeviation - The standard deviation of the
     * normal distribution (mean 0) used to initialize the linear output
     * layer's weight matrix. Defaults to 0.02, as in GPT-2. Must be a finite
     * number greater than or equal to 0.
     * @param outputLength - The number of tokens the model should generate.
     * Defaults to 2. Must be a positive whole number.
     * @param seed - Optional PRNG seed for the random number stream returned
     * by random(), which every component uses for weight initialization,
     * dropout and decoding. When provided, the same seed always produces the
     * same results, provided components are created and called in the same
     * order. When omitted, Math.random() is used and results differ on every
     * run.
     * @throws RangeError if dropoutRate is less than 0, greater than 1 or NaN.
     * @throws RangeError if numberAttentionHeads is not a positive whole number.
     * @throws RangeError if embeddingSize divided by numberAttentionHeads is
     * not a whole number.
     * @throws RangeError if numberTransformerBlocks is not a positive whole number.
     * @throws RangeError if batchSize is not a positive whole number.
     * @throws RangeError if positionEmbeddingStandardDeviation,
     * tokenEmbeddingStandardDeviation, attentionProjectionStandardDeviation or
     * outputLayerStandardDeviation is negative, infinite or NaN.
     * @throws RangeError if outputLength is not a positive whole number.
     */
    constructor(embeddingSize = 768, contextLength = 12, //GPT-2 uses 1024
    dropoutRate = 0.1, training = true, numberAttentionHeads = 12, numberTransformerBlocks = 12, batchSize = 2, qkvBias = false, feedForwardBias = true, positionEmbeddingStandardDeviation = 0.01, tokenEmbeddingStandardDeviation = 0.02, attentionProjectionStandardDeviation = 0.02, outputLayerStandardDeviation = 0.02, outputLength = 2, seed) {
        Hyperparameters.validateDropoutRate(dropoutRate);
        Hyperparameters.validatePositiveInteger("numberAttentionHeads", numberAttentionHeads);
        const weightMatrixColumns = Hyperparameters.computeWeightMatrixColumns(embeddingSize, numberAttentionHeads);
        Hyperparameters.validatePositiveInteger("numberTransformerBlocks", numberTransformerBlocks);
        Hyperparameters.validatePositiveInteger("batchSize", batchSize);
        Hyperparameters.validateStandardDeviation("positionEmbeddingStandardDeviation", positionEmbeddingStandardDeviation);
        Hyperparameters.validateStandardDeviation("tokenEmbeddingStandardDeviation", tokenEmbeddingStandardDeviation);
        Hyperparameters.validateStandardDeviation("attentionProjectionStandardDeviation", attentionProjectionStandardDeviation);
        Hyperparameters.validateStandardDeviation("outputLayerStandardDeviation", outputLayerStandardDeviation);
        Hyperparameters.validatePositiveInteger("outputLength", outputLength);
        this._embeddingSize = embeddingSize;
        this._contextLength = contextLength;
        this._vocabularySize = new BPETokenizer().vocabularySize;
        this._weightMatrixColumns = weightMatrixColumns;
        this._dropoutRate = dropoutRate;
        this._training = training;
        this._numberAttentionHeads = numberAttentionHeads;
        this._numberTransformerBlocks = numberTransformerBlocks;
        this._batchSize = batchSize;
        this._qkvBias = qkvBias;
        this._feedForwardBias = feedForwardBias;
        this._positionEmbeddingStandardDeviation = positionEmbeddingStandardDeviation;
        this._tokenEmbeddingStandardDeviation = tokenEmbeddingStandardDeviation;
        this._attentionProjectionStandardDeviation = attentionProjectionStandardDeviation;
        this._outputLayerStandardDeviation = outputLayerStandardDeviation;
        this._outputLength = outputLength;
        this._seed = seed;
        this._random = Hyperparameters.createRandom(seed);
    }
    /**
     * Creates the random number stream for a seed.
     *
     * @param seed - The seed, or undefined for an unseeded stream.
     * @returns A seeded PRNG when seed is set; otherwise Math.random.
     */
    static createRandom(seed) {
        return seed === undefined ? Math.random : createSeededRandom(seed);
    }
    /**
     * Checks that a dropout rate lies between 0 and 1 inclusive.
     *
     * @param dropoutRate - The dropout rate to check.
     * @throws RangeError if dropoutRate is less than 0, greater than 1 or NaN.
     */
    static validateDropoutRate(dropoutRate) {
        // Written as a negated range check so that NaN, which fails every
        // comparison, is rejected along with values outside [0, 1].
        if (!(dropoutRate >= 0 && dropoutRate <= 1)) {
            throw new RangeError(`Hyperparameters| dropoutRate must be between 0 and 1; got ${dropoutRate}`);
        }
    }
    /**
     * Checks that a value is a positive whole number.
     *
     * @param name - The name of the hyperparameter, used in the error message.
     * @param value - The value to check.
     * @throws RangeError if value is not a positive whole number.
     */
    static validatePositiveInteger(name, value) {
        // Number.isInteger rejects fractions, NaN and Infinity, so combined
        // with the > 0 check only positive whole numbers are accepted.
        if (!(Number.isInteger(value) && value > 0)) {
            throw new RangeError(`Hyperparameters| ${name} must be a positive whole number; got ${value}`);
        }
    }
    /**
     * Computes the number of columns in each query/key/value weight matrix
     * and checks that the attention heads split the embedding evenly.
     *
     * @param embeddingSize - The size of each token embedding vector.
     * @param numberAttentionHeads - The number of attention heads.
     * @returns embeddingSize / numberAttentionHeads.
     * @throws RangeError if embeddingSize divided by numberAttentionHeads is
     * not a whole number.
     */
    static computeWeightMatrixColumns(embeddingSize, numberAttentionHeads) {
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
        return weightMatrixColumns;
    }
    /**
     * Checks that a weight initialization standard deviation is a finite
     * number greater than or equal to 0.
     *
     * @param name - The name of the hyperparameter, used in the error message.
     * @param standardDeviation - The standard deviation to check.
     * @throws RangeError if standardDeviation is negative, infinite or NaN.
     */
    static validateStandardDeviation(name, standardDeviation) {
        // Number.isFinite rejects NaN and Infinity; a negative standard deviation is meaningless.
        if (!(Number.isFinite(standardDeviation) && standardDeviation >= 0)) {
            throw new RangeError(`Hyperparameters| ${name} must be a finite number >= 0; got ${standardDeviation}`);
        }
    }
    /** The size of each token embedding vector. */
    get embeddingSize() {
        return this._embeddingSize;
    }
    /**
     * Sets the embedding size and recomputes weightMatrixColumns.
     *
     * @throws RangeError if the new embeddingSize divided by
     * numberAttentionHeads is not a whole number.
     */
    set embeddingSize(embeddingSize) {
        this._weightMatrixColumns = Hyperparameters.computeWeightMatrixColumns(embeddingSize, this._numberAttentionHeads);
        this._embeddingSize = embeddingSize;
    }
    /** The number of tokens of context the model operates over at once. */
    get contextLength() {
        return this._contextLength;
    }
    set contextLength(contextLength) {
        this._contextLength = contextLength;
    }
    /**
     * The number of columns in each of the query/key/value weight matrices,
     * i.e. the size of the projected vectors those matrices produce. Equal to
     * embeddingSize / numberAttentionHeads, so the concatenated output of all
     * the attention heads is embeddingSize wide. Derived, so it has no setter.
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
    /** @throws RangeError if dropoutRate is less than 0, greater than 1 or NaN. */
    set dropoutRate(dropoutRate) {
        Hyperparameters.validateDropoutRate(dropoutRate);
        this._dropoutRate = dropoutRate;
    }
    /**
     * True for training mode, in which dropout is applied; false for
     * inference mode, in which dropout is skipped.
     */
    get training() {
        return this._training;
    }
    set training(training) {
        this._training = training;
    }
    /** The number of attention heads used in multi-head attention. */
    get numberAttentionHeads() {
        return this._numberAttentionHeads;
    }
    /**
     * Sets the number of attention heads and recomputes weightMatrixColumns.
     *
     * @throws RangeError if numberAttentionHeads is not a positive whole number.
     * @throws RangeError if embeddingSize divided by the new
     * numberAttentionHeads is not a whole number.
     */
    set numberAttentionHeads(numberAttentionHeads) {
        Hyperparameters.validatePositiveInteger("numberAttentionHeads", numberAttentionHeads);
        this._weightMatrixColumns = Hyperparameters.computeWeightMatrixColumns(this._embeddingSize, numberAttentionHeads);
        this._numberAttentionHeads = numberAttentionHeads;
    }
    /** The number of transformer blocks stacked in the model. */
    get numberTransformerBlocks() {
        return this._numberTransformerBlocks;
    }
    /** @throws RangeError if numberTransformerBlocks is not a positive whole number. */
    set numberTransformerBlocks(numberTransformerBlocks) {
        Hyperparameters.validatePositiveInteger("numberTransformerBlocks", numberTransformerBlocks);
        this._numberTransformerBlocks = numberTransformerBlocks;
    }
    /** The number of input sequences processed together in one batch. */
    get batchSize() {
        return this._batchSize;
    }
    /** @throws RangeError if batchSize is not a positive whole number. */
    set batchSize(batchSize) {
        Hyperparameters.validatePositiveInteger("batchSize", batchSize);
        this._batchSize = batchSize;
    }
    /**
     * True if the query, key and value projections in each attention head add
     * a bias to their outputs; false if they do not.
     */
    get qkvBias() {
        return this._qkvBias;
    }
    set qkvBias(qkvBias) {
        this._qkvBias = qkvBias;
    }
    /**
     * True if the linear layers of the feed forward network add a bias to
     * their outputs; false if they do not.
     */
    get feedForwardBias() {
        return this._feedForwardBias;
    }
    set feedForwardBias(feedForwardBias) {
        this._feedForwardBias = feedForwardBias;
    }
    /**
     * The standard deviation of the normal distribution (mean 0) used to
     * initialize the position embedding matrix. The original GPT-2 uses 0.01.
     */
    get positionEmbeddingStandardDeviation() {
        return this._positionEmbeddingStandardDeviation;
    }
    /** @throws RangeError if positionEmbeddingStandardDeviation is negative, infinite or NaN. */
    set positionEmbeddingStandardDeviation(positionEmbeddingStandardDeviation) {
        Hyperparameters.validateStandardDeviation("positionEmbeddingStandardDeviation", positionEmbeddingStandardDeviation);
        this._positionEmbeddingStandardDeviation = positionEmbeddingStandardDeviation;
    }
    /**
     * The standard deviation of the normal distribution (mean 0) used to
     * initialize the token embedding matrix. GPT-2 uses 0.02.
     */
    get tokenEmbeddingStandardDeviation() {
        return this._tokenEmbeddingStandardDeviation;
    }
    /** @throws RangeError if tokenEmbeddingStandardDeviation is negative, infinite or NaN. */
    set tokenEmbeddingStandardDeviation(tokenEmbeddingStandardDeviation) {
        Hyperparameters.validateStandardDeviation("tokenEmbeddingStandardDeviation", tokenEmbeddingStandardDeviation);
        this._tokenEmbeddingStandardDeviation = tokenEmbeddingStandardDeviation;
    }
    /**
     * The base standard deviation of the normal distribution (mean 0) used to
     * initialize the multi-head attention output projection. GPT-2 uses 0.02.
     * MultiHeadAttention divides it by sqrt(2 * numberTransformerBlocks) to
     * stop the residual stream's variance growing with the model's depth.
     */
    get attentionProjectionStandardDeviation() {
        return this._attentionProjectionStandardDeviation;
    }
    /** @throws RangeError if attentionProjectionStandardDeviation is negative, infinite or NaN. */
    set attentionProjectionStandardDeviation(attentionProjectionStandardDeviation) {
        Hyperparameters.validateStandardDeviation("attentionProjectionStandardDeviation", attentionProjectionStandardDeviation);
        this._attentionProjectionStandardDeviation = attentionProjectionStandardDeviation;
    }
    /**
     * The standard deviation of the normal distribution (mean 0) used to
     * initialize the linear output layer's weight matrix. GPT-2 uses 0.02.
     */
    get outputLayerStandardDeviation() {
        return this._outputLayerStandardDeviation;
    }
    /** @throws RangeError if outputLayerStandardDeviation is negative, infinite or NaN. */
    set outputLayerStandardDeviation(outputLayerStandardDeviation) {
        Hyperparameters.validateStandardDeviation("outputLayerStandardDeviation", outputLayerStandardDeviation);
        this._outputLayerStandardDeviation = outputLayerStandardDeviation;
    }
    /** The number of tokens the model should generate. */
    get outputLength() {
        return this._outputLength;
    }
    /** @throws RangeError if outputLength is not a positive whole number. */
    set outputLength(outputLength) {
        Hyperparameters.validatePositiveInteger("outputLength", outputLength);
        this._outputLength = outputLength;
    }
    /**
     * Optional PRNG seed for the random number stream returned by random().
     * Undefined means Math.random() is used, so results are not reproducible.
     */
    get seed() {
        return this._seed;
    }
    /** Sets the seed and restarts the random number stream from it. */
    set seed(seed) {
        this._seed = seed;
        this._random = Hyperparameters.createRandom(seed);
    }
    /**
     * Returns the next value from the shared random number stream. Every
     * component draws from this one stream, rather than seeding its own, so
     * components of the same kind (e.g. attention heads) receive different
     * values while the run as a whole stays reproducible for a given seed.
     *
     * @returns A pseudo-random value in [0, 1).
     */
    random() {
        return this._random();
    }
    /**
     * The size of the vocabulary, loaded from BPETokenizer in the
     * constructor so it always matches the tokenizer actually in use.
     * Derived, so it has no setter.
     */
    get vocabularySize() {
        return this._vocabularySize;
    }
}
export default Hyperparameters;
