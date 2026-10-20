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
    /**
     * Mode for training the model: dropout is applied, and the model returns
     * logits for every token so the loss can be calculated at every position.
     */
    static TRAINING_MODE = 0;
    /**
     * Mode for measuring the loss on held-out data: dropout is skipped, so the
     * results are deterministic, but the model still returns logits for every
     * token so the loss can be calculated at every position.
     */
    static VALIDATION_MODE = 1;
    /**
     * Mode for generating text: dropout is skipped, and the model only returns
     * logits for the last token, since only the next token is being predicted.
     */
    static GENERATION_MODE = 2;
    _embeddingSize;
    _contextLength;
    _vocabularySize;
    _weightMatrixColumns;
    _dropoutRate;
    _mode;
    _numberAttentionHeads;
    _numberTransformerBlocks;
    _batchSize;
    _qkvBias;
    _feedForwardBias;
    _positionEmbeddingStandardDeviation;
    _tokenEmbeddingStandardDeviation;
    _linearLayerStandardDeviation;
    _outputLayerStandardDeviation;
    _outputLength;
    _trainingRatio;
    _numberEpochs;
    _learningRate;
    _beta1;
    _beta2;
    _optimizerEpsilon;
    _weightDecay;
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
     * operates over at once. Defaults to 12 to keep runs fast; GPT-2 uses
     * 1024. The position embedding matrix has contextLength rows, so this
     * also changes the parameter count: with the other defaults the model has
     * 163,009,536 parameters (matching the book) only when contextLength is
     * 1024, and 162,232,320 with the default of 12. Must be a positive whole
     * number.
     * @param dropoutRate - The probability, between 0 and 1 inclusive, that
     * any given value is dropped (set to zero) during dropout.
     * @param mode - What the model is being used for: TRAINING_MODE,
     * VALIDATION_MODE or GENERATION_MODE. Defaults to TRAINING_MODE.
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
     * @param linearLayerStandardDeviation - The standard deviation of the
     * normal distribution (mean 0) used to initialize the weights of every
     * linear layer in the transformer blocks: the query, key and value
     * weights, the attention output projection and both feed forward layers.
     * The two residual projections (the attention output projection and the
     * second feed forward layer) divide it by sqrt(2 * numberTransformerBlocks).
     * Defaults to 0.02, GPT-2's initializer_range. Must be a finite number
     * greater than or equal to 0.
     * @param outputLayerStandardDeviation - The standard deviation of the
     * normal distribution (mean 0) used to initialize the linear output
     * layer's weight matrix. Defaults to 0.02, as in GPT-2. Must be a finite
     * number greater than or equal to 0.
     * @param outputLength - The number of tokens the model should generate.
     * Defaults to 2. Must be a positive whole number.
     * @param trainingRatio - The proportion of the loaded text used for
     * training; the rest is held back for validation. Defaults to 0.9, as in
     * the book. Must be greater than 0 and less than 1, so that neither set
     * is empty.
     * @param numberEpochs - The number of times training passes over the whole
     * training set. Defaults to 1. The book uses 10, but until the weights are
     * updated every epoch gives the same loss, so further epochs only take
     * time. Must be a positive whole number.
     * @param learningRate - The size of each AdamW weight update. Defaults to
     * 0.0004, the value the book uses in chapter 5 (dnn uses 0.001). Must be a
     * finite number greater than 0.
     * @param beta1 - How much of the previous running average of the gradients
     * AdamW keeps at each step (its momentum). Defaults to 0.9. Must be at
     * least 0 and less than 1.
     * @param beta2 - How much of the previous running average of the squared
     * gradients AdamW keeps at each step. Defaults to 0.999. Must be at least 0
     * and less than 1.
     * @param optimizerEpsilon - A small number AdamW adds before dividing, so
     * it never divides by zero. Defaults to 1e-8. Not to be confused with
     * LayerNormalization.epsilon. Must be a finite number greater than 0.
     * @param weightDecay - How strongly AdamW pulls the weights towards 0 at
     * each step, which discourages large weights and so reduces overfitting.
     * Defaults to 0.1, the book's value (dnn uses 0.01). Must be a finite
     * number greater than or equal to 0.
     * @param seed - Optional PRNG seed for the random number stream returned
     * by random(), which every component uses for weight initialization,
     * dropout and decoding. When provided, the same seed always produces the
     * same results, provided components are created and called in the same
     * order. When omitted, Math.random() is used and results differ on every
     * run.
     * @throws RangeError if contextLength is not a positive whole number.
     * @throws RangeError if dropoutRate is less than 0, greater than 1 or NaN.
     * @throws RangeError if mode is not TRAINING_MODE, VALIDATION_MODE or GENERATION_MODE.
     * @throws RangeError if numberAttentionHeads is not a positive whole number.
     * @throws RangeError if embeddingSize divided by numberAttentionHeads is
     * not a whole number.
     * @throws RangeError if numberTransformerBlocks is not a positive whole number.
     * @throws RangeError if batchSize is not a positive whole number.
     * @throws RangeError if positionEmbeddingStandardDeviation,
     * tokenEmbeddingStandardDeviation, linearLayerStandardDeviation or
     * outputLayerStandardDeviation is negative, infinite or NaN.
     * @throws RangeError if outputLength is not a positive whole number.
     * @throws RangeError if trainingRatio is not greater than 0 and less than 1.
     * @throws RangeError if numberEpochs is not a positive whole number.
     * @throws RangeError if learningRate or optimizerEpsilon is not a finite number greater than 0.
     * @throws RangeError if beta1 or beta2 is not at least 0 and less than 1.
     * @throws RangeError if weightDecay is negative, infinite or NaN.
     */
    constructor(embeddingSize = 768, contextLength = 12, //GPT-2 uses 1024
    dropoutRate = 0.1, mode = Hyperparameters.TRAINING_MODE, numberAttentionHeads = 12, numberTransformerBlocks = 12, batchSize = 2, qkvBias = false, feedForwardBias = true, positionEmbeddingStandardDeviation = 0.01, tokenEmbeddingStandardDeviation = 0.02, linearLayerStandardDeviation = 0.02, outputLayerStandardDeviation = 0.02, outputLength = 2, trainingRatio = 0.9, numberEpochs = 1, learningRate = 0.0004, beta1 = 0.9, beta2 = 0.999, optimizerEpsilon = 1e-8, weightDecay = 0.1, seed) {
        Hyperparameters.validatePositiveInteger("contextLength", contextLength);
        Hyperparameters.validateDropoutRate(dropoutRate);
        Hyperparameters.validateMode(mode);
        Hyperparameters.validatePositiveInteger("numberAttentionHeads", numberAttentionHeads);
        const weightMatrixColumns = Hyperparameters.computeWeightMatrixColumns(embeddingSize, numberAttentionHeads);
        Hyperparameters.validatePositiveInteger("numberTransformerBlocks", numberTransformerBlocks);
        Hyperparameters.validatePositiveInteger("batchSize", batchSize);
        Hyperparameters.validateStandardDeviation("positionEmbeddingStandardDeviation", positionEmbeddingStandardDeviation);
        Hyperparameters.validateStandardDeviation("tokenEmbeddingStandardDeviation", tokenEmbeddingStandardDeviation);
        Hyperparameters.validateStandardDeviation("linearLayerStandardDeviation", linearLayerStandardDeviation);
        Hyperparameters.validateStandardDeviation("outputLayerStandardDeviation", outputLayerStandardDeviation);
        Hyperparameters.validatePositiveInteger("outputLength", outputLength);
        Hyperparameters.validateTrainingRatio(trainingRatio);
        Hyperparameters.validatePositiveInteger("numberEpochs", numberEpochs);
        Hyperparameters.validatePositiveNumber("learningRate", learningRate);
        Hyperparameters.validateBeta("beta1", beta1);
        Hyperparameters.validateBeta("beta2", beta2);
        Hyperparameters.validatePositiveNumber("optimizerEpsilon", optimizerEpsilon);
        Hyperparameters.validateWeightDecay(weightDecay);
        this._embeddingSize = embeddingSize;
        this._contextLength = contextLength;
        this._vocabularySize = new BPETokenizer().vocabularySize;
        this._weightMatrixColumns = weightMatrixColumns;
        this._dropoutRate = dropoutRate;
        this._mode = mode;
        this._numberAttentionHeads = numberAttentionHeads;
        this._numberTransformerBlocks = numberTransformerBlocks;
        this._batchSize = batchSize;
        this._qkvBias = qkvBias;
        this._feedForwardBias = feedForwardBias;
        this._positionEmbeddingStandardDeviation = positionEmbeddingStandardDeviation;
        this._tokenEmbeddingStandardDeviation = tokenEmbeddingStandardDeviation;
        this._linearLayerStandardDeviation = linearLayerStandardDeviation;
        this._outputLayerStandardDeviation = outputLayerStandardDeviation;
        this._outputLength = outputLength;
        this._trainingRatio = trainingRatio;
        this._numberEpochs = numberEpochs;
        this._learningRate = learningRate;
        this._beta1 = beta1;
        this._beta2 = beta2;
        this._optimizerEpsilon = optimizerEpsilon;
        this._weightDecay = weightDecay;
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
     * Checks that a mode is one of the three defined modes.
     *
     * @param mode - The mode to check.
     * @throws RangeError if mode is not TRAINING_MODE, VALIDATION_MODE or GENERATION_MODE.
     */
    static validateMode(mode) {
        const modes = [
            Hyperparameters.TRAINING_MODE,
            Hyperparameters.VALIDATION_MODE,
            Hyperparameters.GENERATION_MODE,
        ];
        if (!modes.includes(mode)) {
            throw new RangeError(`Hyperparameters| mode must be TRAINING_MODE (${Hyperparameters.TRAINING_MODE}), ` +
                `VALIDATION_MODE (${Hyperparameters.VALIDATION_MODE}) or ` +
                `GENERATION_MODE (${Hyperparameters.GENERATION_MODE}); got ${mode}`);
        }
    }
    /**
     * Checks that a training ratio leaves something in both the training and
     * the validation set.
     *
     * @param trainingRatio - The training ratio to check.
     * @throws RangeError if trainingRatio is not greater than 0 and less than 1, or is NaN.
     */
    static validateTrainingRatio(trainingRatio) {
        // Written as a negated range check so that NaN, which fails every comparison, is rejected.
        if (!(trainingRatio > 0 && trainingRatio < 1)) {
            throw new RangeError(`Hyperparameters| trainingRatio must be greater than 0 and less than 1; got ${trainingRatio}`);
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
    /**
     * Checks that a value is a finite number greater than 0.
     *
     * @param name - The name of the hyperparameter, used in the error message.
     * @param value - The value to check.
     * @throws RangeError if value is 0 or less, infinite or NaN.
     */
    static validatePositiveNumber(name, value) {
        // Number.isFinite rejects NaN and Infinity.
        if (!(Number.isFinite(value) && value > 0)) {
            throw new RangeError(`Hyperparameters| ${name} must be a finite number > 0; got ${value}`);
        }
    }
    /**
     * Checks that an AdamW running-average rate (beta1 or beta2) is at least 0
     * and less than 1. A value of 1 would never let the average change, and
     * AdamW's bias correction would divide by zero.
     *
     * @param name - The name of the hyperparameter, used in the error message.
     * @param value - The value to check.
     * @throws RangeError if value is less than 0, 1 or more, or NaN.
     */
    static validateBeta(name, value) {
        // Written as a negated range check so that NaN, which fails every comparison, is rejected.
        if (!(value >= 0 && value < 1)) {
            throw new RangeError(`Hyperparameters| ${name} must be at least 0 and less than 1; got ${value}`);
        }
    }
    /**
     * Checks that a weight decay is a finite number greater than or equal to 0.
     *
     * @param weightDecay - The weight decay to check.
     * @throws RangeError if weightDecay is negative, infinite or NaN.
     */
    static validateWeightDecay(weightDecay) {
        if (!(Number.isFinite(weightDecay) && weightDecay >= 0)) {
            throw new RangeError(`Hyperparameters| weightDecay must be a finite number >= 0; got ${weightDecay}`);
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
    /** @throws RangeError if contextLength is not a positive whole number. */
    set contextLength(contextLength) {
        Hyperparameters.validatePositiveInteger("contextLength", contextLength);
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
     * What the model is being used for: TRAINING_MODE (dropout applied,
     * logits for every token), VALIDATION_MODE (no dropout, logits for every
     * token) or GENERATION_MODE (no dropout, logits for the last token only).
     * Components read this on every use, so changing it after they are built
     * takes effect immediately.
     */
    get mode() {
        return this._mode;
    }
    /** @throws RangeError if mode is not TRAINING_MODE, VALIDATION_MODE or GENERATION_MODE. */
    set mode(mode) {
        Hyperparameters.validateMode(mode);
        this._mode = mode;
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
     * The standard deviation of the normal distribution (mean 0) used to
     * initialize the weights of every linear layer in the transformer blocks.
     * GPT-2 uses 0.02. The two residual projections (MultiHeadAttention's
     * output projection and FeedForward's second layer) divide it by
     * sqrt(2 * numberTransformerBlocks) to stop the residual stream's variance
     * growing with the model's depth.
     */
    get linearLayerStandardDeviation() {
        return this._linearLayerStandardDeviation;
    }
    /** @throws RangeError if linearLayerStandardDeviation is negative, infinite or NaN. */
    set linearLayerStandardDeviation(linearLayerStandardDeviation) {
        Hyperparameters.validateStandardDeviation("linearLayerStandardDeviation", linearLayerStandardDeviation);
        this._linearLayerStandardDeviation = linearLayerStandardDeviation;
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
     * The proportion of the loaded text used for training; the rest is held
     * back for validation.
     */
    get trainingRatio() {
        return this._trainingRatio;
    }
    /** @throws RangeError if trainingRatio is not greater than 0 and less than 1. */
    set trainingRatio(trainingRatio) {
        Hyperparameters.validateTrainingRatio(trainingRatio);
        this._trainingRatio = trainingRatio;
    }
    /** The number of times training passes over the whole training set. */
    get numberEpochs() {
        return this._numberEpochs;
    }
    /** @throws RangeError if numberEpochs is not a positive whole number. */
    set numberEpochs(numberEpochs) {
        Hyperparameters.validatePositiveInteger("numberEpochs", numberEpochs);
        this._numberEpochs = numberEpochs;
    }
    /** The size of each AdamW weight update. AdamW reads it on every update, so changing it takes effect at the next batch. */
    get learningRate() {
        return this._learningRate;
    }
    /** @throws RangeError if learningRate is not a finite number greater than 0. */
    set learningRate(learningRate) {
        Hyperparameters.validatePositiveNumber("learningRate", learningRate);
        this._learningRate = learningRate;
    }
    /** How much of the previous running average of the gradients AdamW keeps at each step (its momentum). */
    get beta1() {
        return this._beta1;
    }
    /** @throws RangeError if beta1 is not at least 0 and less than 1. */
    set beta1(beta1) {
        Hyperparameters.validateBeta("beta1", beta1);
        this._beta1 = beta1;
    }
    /** How much of the previous running average of the squared gradients AdamW keeps at each step. */
    get beta2() {
        return this._beta2;
    }
    /** @throws RangeError if beta2 is not at least 0 and less than 1. */
    set beta2(beta2) {
        Hyperparameters.validateBeta("beta2", beta2);
        this._beta2 = beta2;
    }
    /** A small number AdamW adds before dividing, so it never divides by zero. Not LayerNormalization.epsilon. */
    get optimizerEpsilon() {
        return this._optimizerEpsilon;
    }
    /** @throws RangeError if optimizerEpsilon is not a finite number greater than 0. */
    set optimizerEpsilon(optimizerEpsilon) {
        Hyperparameters.validatePositiveNumber("optimizerEpsilon", optimizerEpsilon);
        this._optimizerEpsilon = optimizerEpsilon;
    }
    /** How strongly AdamW pulls the weights towards 0 at each step. 0 turns weight decay off. */
    get weightDecay() {
        return this._weightDecay;
    }
    /** @throws RangeError if weightDecay is negative, infinite or NaN. */
    set weightDecay(weightDecay) {
        Hyperparameters.validateWeightDecay(weightDecay);
        this._weightDecay = weightDecay;
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
