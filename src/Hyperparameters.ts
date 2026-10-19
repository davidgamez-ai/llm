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
  static readonly TRAINING_MODE: number = 0;

  /**
   * Mode for measuring the loss on held-out data: dropout is skipped, so the
   * results are deterministic, but the model still returns logits for every
   * token so the loss can be calculated at every position.
   */
  static readonly VALIDATION_MODE: number = 1;

  /**
   * Mode for generating text: dropout is skipped, and the model only returns
   * logits for the last token, since only the next token is being predicted.
   */
  static readonly GENERATION_MODE: number = 2;

  private _embeddingSize: number;
  private _contextLength: number;
  private readonly _vocabularySize: number;
  private _weightMatrixColumns: number;
  private _dropoutRate: number;
  private _mode: number;
  private _numberAttentionHeads: number;
  private _numberTransformerBlocks: number;
  private _batchSize: number;
  private _qkvBias: boolean;
  private _feedForwardBias: boolean;
  private _positionEmbeddingStandardDeviation: number;
  private _tokenEmbeddingStandardDeviation: number;
  private _linearLayerStandardDeviation: number;
  private _outputLayerStandardDeviation: number;
  private _outputLength: number;
  private _trainingRatio: number;
  private _numberEpochs: number;
  private _seed: number | undefined;

  /**
   * The random number stream shared by every component built from this
   * instance: seeded from _seed when it is set, otherwise Math.random.
   * Replaced whenever the seed is set, so the stream restarts from the
   * beginning.
   */
  private _random: () => number;

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
   */
  constructor(
    embeddingSize: number = 768,
    contextLength: number = 12,//GPT-2 uses 1024
    dropoutRate: number = 0.1,
    mode: number = Hyperparameters.TRAINING_MODE,
    numberAttentionHeads: number = 12,
    numberTransformerBlocks: number = 12,
    batchSize: number = 2,
    qkvBias: boolean = false,
    feedForwardBias: boolean = true,
    positionEmbeddingStandardDeviation: number = 0.01,
    tokenEmbeddingStandardDeviation: number = 0.02,
    linearLayerStandardDeviation: number = 0.02,
    outputLayerStandardDeviation: number = 0.02,
    outputLength: number = 2,
    trainingRatio: number = 0.9,
    numberEpochs: number = 1,
    seed?: number
  ) {
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
    this._seed = seed;
    this._random = Hyperparameters.createRandom(seed);
  }

  /**
   * Creates the random number stream for a seed.
   *
   * @param seed - The seed, or undefined for an unseeded stream.
   * @returns A seeded PRNG when seed is set; otherwise Math.random.
   */
  private static createRandom(seed: number | undefined): () => number {
    return seed === undefined ? Math.random : createSeededRandom(seed);
  }

  /**
   * Checks that a dropout rate lies between 0 and 1 inclusive.
   *
   * @param dropoutRate - The dropout rate to check.
   * @throws RangeError if dropoutRate is less than 0, greater than 1 or NaN.
   */
  private static validateDropoutRate(dropoutRate: number): void {
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
  private static validateMode(mode: number): void {
    const modes: number[] = [
      Hyperparameters.TRAINING_MODE,
      Hyperparameters.VALIDATION_MODE,
      Hyperparameters.GENERATION_MODE,
    ];
    if (!modes.includes(mode)) {
      throw new RangeError(
        `Hyperparameters| mode must be TRAINING_MODE (${Hyperparameters.TRAINING_MODE}), ` +
        `VALIDATION_MODE (${Hyperparameters.VALIDATION_MODE}) or ` +
        `GENERATION_MODE (${Hyperparameters.GENERATION_MODE}); got ${mode}`
      );
    }
  }

  /**
   * Checks that a training ratio leaves something in both the training and
   * the validation set.
   *
   * @param trainingRatio - The training ratio to check.
   * @throws RangeError if trainingRatio is not greater than 0 and less than 1, or is NaN.
   */
  private static validateTrainingRatio(trainingRatio: number): void {
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
  private static validatePositiveInteger(name: string, value: number): void {
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
  private static computeWeightMatrixColumns(embeddingSize: number, numberAttentionHeads: number): number {
    /*
     * Each head produces context vectors weightMatrixColumns wide, and
     * MultiHeadAttention concatenates the heads' outputs, so the combined
     * width only equals embeddingSize if the heads split it evenly.
     */
    const weightMatrixColumns = embeddingSize / numberAttentionHeads;
    if (!Number.isInteger(weightMatrixColumns)) {
      throw new RangeError(
        `Hyperparameters| embeddingSize (${embeddingSize}) divided by numberAttentionHeads ` +
        `(${numberAttentionHeads}) must be a whole number; got ${weightMatrixColumns}`
      );
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
  private static validateStandardDeviation(name: string, standardDeviation: number): void {
    // Number.isFinite rejects NaN and Infinity; a negative standard deviation is meaningless.
    if (!(Number.isFinite(standardDeviation) && standardDeviation >= 0)) {
      throw new RangeError(`Hyperparameters| ${name} must be a finite number >= 0; got ${standardDeviation}`);
    }
  }

  /** The size of each token embedding vector. */
  get embeddingSize(): number {
    return this._embeddingSize;
  }

  /**
   * Sets the embedding size and recomputes weightMatrixColumns.
   *
   * @throws RangeError if the new embeddingSize divided by
   * numberAttentionHeads is not a whole number.
   */
  set embeddingSize(embeddingSize: number) {
    this._weightMatrixColumns = Hyperparameters.computeWeightMatrixColumns(embeddingSize, this._numberAttentionHeads);
    this._embeddingSize = embeddingSize;
  }

  /** The number of tokens of context the model operates over at once. */
  get contextLength(): number {
    return this._contextLength;
  }

  /** @throws RangeError if contextLength is not a positive whole number. */
  set contextLength(contextLength: number) {
    Hyperparameters.validatePositiveInteger("contextLength", contextLength);
    this._contextLength = contextLength;
  }

  /**
   * The number of columns in each of the query/key/value weight matrices,
   * i.e. the size of the projected vectors those matrices produce. Equal to
   * embeddingSize / numberAttentionHeads, so the concatenated output of all
   * the attention heads is embeddingSize wide. Derived, so it has no setter.
   */
  get weightMatrixColumns(): number {
    return this._weightMatrixColumns;
  }

  /**
   * The probability, between 0 and 1 inclusive, that any given value is
   * dropped (set to zero) during dropout.
   */
  get dropoutRate(): number {
    return this._dropoutRate;
  }

  /** @throws RangeError if dropoutRate is less than 0, greater than 1 or NaN. */
  set dropoutRate(dropoutRate: number) {
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
  get mode(): number {
    return this._mode;
  }

  /** @throws RangeError if mode is not TRAINING_MODE, VALIDATION_MODE or GENERATION_MODE. */
  set mode(mode: number) {
    Hyperparameters.validateMode(mode);
    this._mode = mode;
  }

  /** The number of attention heads used in multi-head attention. */
  get numberAttentionHeads(): number {
    return this._numberAttentionHeads;
  }

  /**
   * Sets the number of attention heads and recomputes weightMatrixColumns.
   *
   * @throws RangeError if numberAttentionHeads is not a positive whole number.
   * @throws RangeError if embeddingSize divided by the new
   * numberAttentionHeads is not a whole number.
   */
  set numberAttentionHeads(numberAttentionHeads: number) {
    Hyperparameters.validatePositiveInteger("numberAttentionHeads", numberAttentionHeads);
    this._weightMatrixColumns = Hyperparameters.computeWeightMatrixColumns(this._embeddingSize, numberAttentionHeads);
    this._numberAttentionHeads = numberAttentionHeads;
  }

  /** The number of transformer blocks stacked in the model. */
  get numberTransformerBlocks(): number {
    return this._numberTransformerBlocks;
  }

  /** @throws RangeError if numberTransformerBlocks is not a positive whole number. */
  set numberTransformerBlocks(numberTransformerBlocks: number) {
    Hyperparameters.validatePositiveInteger("numberTransformerBlocks", numberTransformerBlocks);
    this._numberTransformerBlocks = numberTransformerBlocks;
  }

  /** The number of input sequences processed together in one batch. */
  get batchSize(): number {
    return this._batchSize;
  }

  /** @throws RangeError if batchSize is not a positive whole number. */
  set batchSize(batchSize: number) {
    Hyperparameters.validatePositiveInteger("batchSize", batchSize);
    this._batchSize = batchSize;
  }

  /**
   * True if the query, key and value projections in each attention head add
   * a bias to their outputs; false if they do not.
   */
  get qkvBias(): boolean {
    return this._qkvBias;
  }

  set qkvBias(qkvBias: boolean) {
    this._qkvBias = qkvBias;
  }

  /**
   * True if the linear layers of the feed forward network add a bias to
   * their outputs; false if they do not.
   */
  get feedForwardBias(): boolean {
    return this._feedForwardBias;
  }

  set feedForwardBias(feedForwardBias: boolean) {
    this._feedForwardBias = feedForwardBias;
  }

  /**
   * The standard deviation of the normal distribution (mean 0) used to
   * initialize the position embedding matrix. The original GPT-2 uses 0.01.
   */
  get positionEmbeddingStandardDeviation(): number {
    return this._positionEmbeddingStandardDeviation;
  }

  /** @throws RangeError if positionEmbeddingStandardDeviation is negative, infinite or NaN. */
  set positionEmbeddingStandardDeviation(positionEmbeddingStandardDeviation: number) {
    Hyperparameters.validateStandardDeviation("positionEmbeddingStandardDeviation", positionEmbeddingStandardDeviation);
    this._positionEmbeddingStandardDeviation = positionEmbeddingStandardDeviation;
  }

  /**
   * The standard deviation of the normal distribution (mean 0) used to
   * initialize the token embedding matrix. GPT-2 uses 0.02.
   */
  get tokenEmbeddingStandardDeviation(): number {
    return this._tokenEmbeddingStandardDeviation;
  }

  /** @throws RangeError if tokenEmbeddingStandardDeviation is negative, infinite or NaN. */
  set tokenEmbeddingStandardDeviation(tokenEmbeddingStandardDeviation: number) {
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
  get linearLayerStandardDeviation(): number {
    return this._linearLayerStandardDeviation;
  }

  /** @throws RangeError if linearLayerStandardDeviation is negative, infinite or NaN. */
  set linearLayerStandardDeviation(linearLayerStandardDeviation: number) {
    Hyperparameters.validateStandardDeviation("linearLayerStandardDeviation", linearLayerStandardDeviation);
    this._linearLayerStandardDeviation = linearLayerStandardDeviation;
  }

  /**
   * The standard deviation of the normal distribution (mean 0) used to
   * initialize the linear output layer's weight matrix. GPT-2 uses 0.02.
   */
  get outputLayerStandardDeviation(): number {
    return this._outputLayerStandardDeviation;
  }

  /** @throws RangeError if outputLayerStandardDeviation is negative, infinite or NaN. */
  set outputLayerStandardDeviation(outputLayerStandardDeviation: number) {
    Hyperparameters.validateStandardDeviation("outputLayerStandardDeviation", outputLayerStandardDeviation);
    this._outputLayerStandardDeviation = outputLayerStandardDeviation;
  }

  /** The number of tokens the model should generate. */
  get outputLength(): number {
    return this._outputLength;
  }

  /** @throws RangeError if outputLength is not a positive whole number. */
  set outputLength(outputLength: number) {
    Hyperparameters.validatePositiveInteger("outputLength", outputLength);
    this._outputLength = outputLength;
  }

  /**
   * The proportion of the loaded text used for training; the rest is held
   * back for validation.
   */
  get trainingRatio(): number {
    return this._trainingRatio;
  }

  /** @throws RangeError if trainingRatio is not greater than 0 and less than 1. */
  set trainingRatio(trainingRatio: number) {
    Hyperparameters.validateTrainingRatio(trainingRatio);
    this._trainingRatio = trainingRatio;
  }

  /** The number of times training passes over the whole training set. */
  get numberEpochs(): number {
    return this._numberEpochs;
  }

  /** @throws RangeError if numberEpochs is not a positive whole number. */
  set numberEpochs(numberEpochs: number) {
    Hyperparameters.validatePositiveInteger("numberEpochs", numberEpochs);
    this._numberEpochs = numberEpochs;
  }

  /**
   * Optional PRNG seed for the random number stream returned by random().
   * Undefined means Math.random() is used, so results are not reproducible.
   */
  get seed(): number | undefined {
    return this._seed;
  }

  /** Sets the seed and restarts the random number stream from it. */
  set seed(seed: number | undefined) {
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
  random(): number {
    return this._random();
  }

  /**
   * The size of the vocabulary, loaded from BPETokenizer in the
   * constructor so it always matches the tokenizer actually in use.
   * Derived, so it has no setter.
   */
  get vocabularySize(): number {
    return this._vocabularySize;
  }
}

export default Hyperparameters;
