import BPETokenizer from "./BPETokenizer.js";
import createSeededRandom from "./SeededRandom.js";

/**
 * The settable values of a Hyperparameters instance, by their public names,
 * as produced by Hyperparameters.toJSON() and read by fromJSON(). The
 * derived values (vocabularySize and weightMatrixColumns) are left out, since
 * they are calculated from the others.
 */
export interface HyperparameterValues {
  embeddingSize: number;
  contextLength: number;
  dropoutRate: number;
  mode: number;
  numberAttentionHeads: number;
  numberTransformerBlocks: number;
  batchSize: number;
  qkvBias: boolean;
  feedForwardBias: boolean;
  positionEmbeddingStandardDeviation: number;
  tokenEmbeddingStandardDeviation: number;
  linearLayerStandardDeviation: number;
  outputLayerStandardDeviation: number;
  outputLength: number;
  trainingRatio: number;
  numberEpochs: number;
  learningRate: number;
  beta1: number;
  beta2: number;
  optimizerEpsilon: number;
  weightDecay: number;
  numberTrainingExamples: number;
  numberValidationExamples: number;
  temperature: number;
  samplingMethod: number;
  topK: number;
  topP: number;
  /** Left out when there is no seed. */
  seed?: number;
  /** Left out when it is not set. */
  trainingWindowLength?: number;
}

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

  /** Sampling method that always picks the most likely next token. */
  static readonly GREEDY_SAMPLING: number = 0;

  /** Sampling method that picks at random from the k most likely next tokens. */
  static readonly TOP_K_SAMPLING: number = 1;

  /**
   * Sampling method that picks at random from the smallest set of most
   * likely next tokens whose probabilities add up to at least p.
   */
  static readonly TOP_P_SAMPLING: number = 2;

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
  private _learningRate: number;
  private _beta1: number;
  private _beta2: number;
  private _optimizerEpsilon: number;
  private _weightDecay: number;
  private _numberTrainingExamples: number;
  private _numberValidationExamples: number;
  private _temperature: number;
  private _samplingMethod: number;
  private _topK: number;
  private _topP: number;
  private _seed: number | undefined;
  private _trainingWindowLength: number | undefined;

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
   * The defaults for the model's architecture are those of GPT-2 small
   * (embeddingSize 768, contextLength 1024, 12 attention heads, 12
   * transformer blocks, dropout 0.1, biases in every linear layer). With
   * them the model has 163,037,184 parameters. GPT-2 itself has 124,439,808,
   * because it ties the linear output layer to the token embedding (the two
   * share one matrix); here they are separate, as in the book, which adds
   * 50,257 x 768 = 38,597,376.
   *
   * @param embeddingSize - The size of each token embedding vector. Defaults
   * to 768, as in GPT-2 small.
   * @param contextLength - The number of tokens of context the model
   * operates over at once. Defaults to 1024, as in GPT-2. The position
   * embedding matrix has contextLength rows, so this also changes the
   * parameter count. Training runs are much faster with shorter windows; to
   * shorten them without changing the model's shape, as a loaded model needs,
   * use trainingWindowLength. Must be a positive whole number.
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
   * Defaults to true, as in GPT-2. The book sets it to false when training
   * from scratch (giving its count of 163,009,536 parameters), and to true
   * when it loads OpenAI's GPT-2 weights.
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
   * @param trainingRatio - The proportion of each file that TextDataset uses
   * for training; the rest is held back for validation. Datasets of separate
   * examples use numberValidationExamples instead. Defaults to 0.9, as in the
   * book. Must be greater than 0 and less than 1, so that neither set is
   * empty.
   * @param numberEpochs - The number of times training passes over the whole
   * training set. Defaults to 1, so a test run finishes quickly; the book
   * uses 10. Must be a positive whole number.
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
   * @param numberTrainingExamples - The number of examples, such as
   * TinyStories stories, to train on each time a dataset made up of separate
   * examples is loaded. Defaults to 1000: TinyStories averages about 210
   * tokens per story, and the first 1000 come to 235,419 tokens, roughly 2
   * hours per epoch with smallModel(). Must be a positive whole number.
   * @param numberValidationExamples - The number of examples from such a
   * dataset to measure the validation loss on. Defaults to 100 (17,266 tokens
   * of TinyStories): enough for a steady measurement, while taking only a few
   * minutes at the end of each epoch. Must be a positive whole number.
   * @param temperature - How random text generation is: the logits are
   * divided by the temperature before the softmax, so values near 0 make the
   * most likely token ever more certain to be chosen, while 1 leaves the
   * model's probabilities unchanged. Defaults to 1. Must be between 0 and 1
   * inclusive.
   * @param samplingMethod - How the next token is chosen during text
   * generation: GREEDY_SAMPLING, TOP_K_SAMPLING or TOP_P_SAMPLING. Defaults
   * to GREEDY_SAMPLING.
   * @param topK - The number of most likely tokens TOP_K_SAMPLING chooses
   * between. Defaults to 40, the value used by OpenAI's GPT-2 sampling
   * script. Must be a positive whole number; values above vocabularySize
   * keep every token.
   * @param topP - The share of the probability TOP_P_SAMPLING keeps: it
   * chooses between the smallest set of most likely tokens whose
   * probabilities add up to at least topP. Defaults to 0.9, a common choice
   * since Holtzman et al. introduced nucleus sampling. Must be greater than 0
   * and at most 1.
   * @param seed - Optional PRNG seed for the random number stream returned
   * by random(), which every component uses for weight initialization,
   * dropout and decoding. When provided, the same seed always produces the
   * same results, provided components are created and called in the same
   * order. When omitted, Math.random() is used and results differ on every
   * run.
   * @param trainingWindowLength - Optional longest window to train on. When
   * omitted, windows are as long as contextLength. See the trainingWindowLength
   * getter. Must be a positive whole number.
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
   * @throws RangeError if numberTrainingExamples or numberValidationExamples is not a positive whole number.
   * @throws RangeError if temperature is less than 0, greater than 1 or NaN.
   * @throws RangeError if samplingMethod is not GREEDY_SAMPLING, TOP_K_SAMPLING or TOP_P_SAMPLING.
   * @throws RangeError if topK is not a positive whole number.
   * @throws RangeError if topP is not greater than 0 and at most 1.
   * @throws RangeError if trainingWindowLength is set and is not a positive whole number.
   */
  constructor(
    embeddingSize: number = 768,//GPT-2 uses 768
    contextLength: number = 1024,//GPT-2 uses 1024
    dropoutRate: number = 0.1,
    mode: number = Hyperparameters.TRAINING_MODE,
    numberAttentionHeads: number = 12,
    numberTransformerBlocks: number = 12,
    batchSize: number = 2,
    qkvBias: boolean = true,
    feedForwardBias: boolean = true,
    positionEmbeddingStandardDeviation: number = 0.01,
    tokenEmbeddingStandardDeviation: number = 0.02,
    linearLayerStandardDeviation: number = 0.02,
    outputLayerStandardDeviation: number = 0.02,
    outputLength: number = 2,
    trainingRatio: number = 0.9,
    numberEpochs: number = 1,
    learningRate: number = 0.0004,
    beta1: number = 0.9,
    beta2: number = 0.999,
    optimizerEpsilon: number = 1e-8,
    weightDecay: number = 0.1,
    numberTrainingExamples: number = 1000,
    numberValidationExamples: number = 100,
    temperature: number = 1,
    samplingMethod: number = Hyperparameters.GREEDY_SAMPLING,
    topK: number = 40,
    topP: number = 0.9,
    seed?: number,
    trainingWindowLength?: number
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
    Hyperparameters.validatePositiveNumber("learningRate", learningRate);
    Hyperparameters.validateBeta("beta1", beta1);
    Hyperparameters.validateBeta("beta2", beta2);
    Hyperparameters.validatePositiveNumber("optimizerEpsilon", optimizerEpsilon);
    Hyperparameters.validateWeightDecay(weightDecay);
    Hyperparameters.validatePositiveInteger("numberTrainingExamples", numberTrainingExamples);
    Hyperparameters.validatePositiveInteger("numberValidationExamples", numberValidationExamples);
    Hyperparameters.validateTemperature(temperature);
    Hyperparameters.validateSamplingMethod(samplingMethod);
    Hyperparameters.validatePositiveInteger("topK", topK);
    Hyperparameters.validateTopP(topP);
    if (trainingWindowLength !== undefined) {
      Hyperparameters.validatePositiveInteger("trainingWindowLength", trainingWindowLength);
    }

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
    this._numberTrainingExamples = numberTrainingExamples;
    this._numberValidationExamples = numberValidationExamples;
    this._temperature = temperature;
    this._samplingMethod = samplingMethod;
    this._topK = topK;
    this._topP = topP;
    this._seed = seed;
    this._trainingWindowLength = trainingWindowLength;
    this._random = Hyperparameters.createRandom(seed);
  }

  /**
   * Creates the Hyperparameters of the small model, for training from scratch
   * in hours rather than days, as a demonstration that the model learns to
   * write English. With about 110,000 training tokens an hour, it can get
   * through a full pass of Tiny Shakespeare or a useful part of TinyStories.
   *
   * The three model sizes form a family, as GPT-2's do, with about 3.5 times
   * as many parameters and a quarter of the speed at each step. The speeds
   * were measured on one desktop computer (Ryzen 7 3800X):
   *
   *     Size    Width  Heads  Blocks  Context  Parameters   Tokens per hour
   *     small     128      4       4      128  13.7 million         110,000
   *     medium    384      6       6      256  49.3 million          27,000
   *     large     768     12      12     1024   163 million           8,000
   *
   * Most of the training time and most of the small model's parameters are
   * in the vocabulary: the output layer and the token embedding each have
   * 50,257 rows of embeddingSize values, so a wider model trains on fewer
   * tokens an hour. A narrower model of width 64 is about twice as fast and
   * did better in runs of about 40 minutes, because it saw more text
   * (validation loss 4.61, against 4.84 at width 128). Width 128 was chosen
   * for the small model because a model this size can make use of longer
   * training, and because the step up to the medium model is then in
   * proportion with the step from medium to large.
   *
   * The other settings, compared with GPT-2 small:
   * - contextLength 128 instead of 1024. Shorter windows give more weight
   *   updates for the same amount of text.
   * - dropoutRate 0 instead of 0.1. Dropout reduces overfitting when the same
   *   text is seen again and again; a single pass over TinyStories repeats
   *   nothing, so it would only slow learning down. Set it to 0.1 when
   *   training for several epochs on a small text.
   * - learningRate 0.001 instead of the book's 0.0004: smaller models
   *   tolerate, and need, larger steps. This is the rate used in the
   *   experiments at this width.
   * - numberEpochs 1, and outputLength 50 so generated samples are long
   *   enough to judge.
   *
   * Everything else (biases, weight initialization, batchSize 2, the other
   * AdamW settings) keeps its default.
   *
   * @returns A new Hyperparameters instance, in TRAINING_MODE, with no seed.
   */
  static smallModel(): Hyperparameters {
    return Hyperparameters.reducedModel(128, 4, 4, 128, 0.001);
  }

  /**
   * Creates the Hyperparameters of the medium model, between the small model
   * and GPT-2 small in both size and speed (see smallModel). At about 27,000
   * training tokens an hour, it suits training over several days.
   *
   * The settings, compared with GPT-2 small:
   * - embeddingSize 384, half of GPT-2 small's width.
   * - numberAttentionHeads 6, so each head works with 64 values, as in GPT-2
   *   small.
   * - numberTransformerBlocks 6, half of GPT-2 small's depth.
   * - contextLength 256, the value the book uses when training. TinyStories
   *   averages about 210 tokens per story, so most stories fit in one window.
   * - learningRate 0.0005, between the small model's rate and the book's
   *   0.0004 for GPT-2 small. This is an estimate, not the result of an
   *   experiment, as no long run of this size has been made.
   * - dropoutRate 0, numberEpochs 1 and outputLength 50, as in the small
   *   model.
   *
   * @returns A new Hyperparameters instance, in TRAINING_MODE, with no seed.
   */
  static mediumModel(): Hyperparameters {
    return Hyperparameters.reducedModel(384, 6, 6, 256, 0.0005);
  }

  /**
   * Creates the Hyperparameters of the large model, which matches GPT-2 small
   * (see smallModel for the family of sizes). These are the constructor's
   * defaults, apart from outputLength: a model this size takes about a minute
   * to generate 25 tokens, so it generates 15 rather than the 50 of the
   * smaller models.
   *
   * At about 8,000 training tokens an hour, it is too slow to train from
   * scratch in plain TypeScript. Use it to continue training from OpenAI's
   * weights (see ModelManager.loadGPT2).
   *
   * @returns A new Hyperparameters instance, in TRAINING_MODE, with no seed.
   */
  static largeModel(): Hyperparameters {
    const hyperparameters = new Hyperparameters();
    hyperparameters.outputLength = 15;
    return hyperparameters;
  }

  /**
   * Creates the Hyperparameters of a model smaller than GPT-2 small, with the
   * training settings shared by smallModel and mediumModel: no dropout, one
   * epoch and outputLength 50.
   *
   * @param embeddingSize - The size of each token embedding vector.
   * @param numberAttentionHeads - The number of attention heads. Must divide embeddingSize exactly.
   * @param numberTransformerBlocks - The number of transformer blocks.
   * @param contextLength - The number of tokens of context.
   * @param learningRate - The size of each AdamW weight update.
   * @returns A new Hyperparameters instance, in TRAINING_MODE, with no seed.
   */
  private static reducedModel(
    embeddingSize: number,
    numberAttentionHeads: number,
    numberTransformerBlocks: number,
    contextLength: number,
    learningRate: number
  ): Hyperparameters {
    const hyperparameters = new Hyperparameters();

    /*
     * The number of heads is set before the embedding size because each
     * setter checks that the embedding divides evenly between the heads:
     * 768 is divisible by the new number of heads, but the new embedding size
     * need not be divisible by the default 12.
     */
    hyperparameters.numberAttentionHeads = numberAttentionHeads;
    hyperparameters.embeddingSize = embeddingSize;
    hyperparameters.numberTransformerBlocks = numberTransformerBlocks;
    hyperparameters.contextLength = contextLength;
    hyperparameters.dropoutRate = 0;
    hyperparameters.learningRate = learningRate;
    hyperparameters.numberEpochs = 1;
    hyperparameters.outputLength = 50;
    return hyperparameters;
  }

  /**
   * Lists the settable values by their public names, so they can be saved,
   * e.g. with a model's weights (see ModelManager). Called automatically by
   * JSON.stringify, which would otherwise write the private fields, with
   * their underscores, and drop the random number stream.
   *
   * @returns Every settable value. seed and trainingWindowLength are left out when they are not set.
   */
  toJSON(): HyperparameterValues {
    return {
      embeddingSize: this._embeddingSize,
      contextLength: this._contextLength,
      dropoutRate: this._dropoutRate,
      mode: this._mode,
      numberAttentionHeads: this._numberAttentionHeads,
      numberTransformerBlocks: this._numberTransformerBlocks,
      batchSize: this._batchSize,
      qkvBias: this._qkvBias,
      feedForwardBias: this._feedForwardBias,
      positionEmbeddingStandardDeviation: this._positionEmbeddingStandardDeviation,
      tokenEmbeddingStandardDeviation: this._tokenEmbeddingStandardDeviation,
      linearLayerStandardDeviation: this._linearLayerStandardDeviation,
      outputLayerStandardDeviation: this._outputLayerStandardDeviation,
      outputLength: this._outputLength,
      trainingRatio: this._trainingRatio,
      numberEpochs: this._numberEpochs,
      learningRate: this._learningRate,
      beta1: this._beta1,
      beta2: this._beta2,
      optimizerEpsilon: this._optimizerEpsilon,
      weightDecay: this._weightDecay,
      numberTrainingExamples: this._numberTrainingExamples,
      numberValidationExamples: this._numberValidationExamples,
      temperature: this._temperature,
      samplingMethod: this._samplingMethod,
      topK: this._topK,
      topP: this._topP,
      // JSON.stringify drops undefined values, so an unset seed or window length is simply left out.
      seed: this._seed,
      // The value as set, rather than the getter's result, so that a window length that is unset stays unset.
      trainingWindowLength: this._trainingWindowLength,
    };
  }

  /**
   * Creates Hyperparameters from values produced by toJSON(), e.g. read back
   * from a saved model.
   *
   * The values are passed to the constructor rather than set one by one, so
   * they are all validated as usual, and the order does not matter: setting
   * numberAttentionHeads and embeddingSize separately could fail part way,
   * since each setter checks that the embedding divides evenly between the
   * heads. Any value that is missing takes its default. The random number
   * stream starts from the beginning of the seed's stream, not from wherever
   * the saved instance had reached.
   *
   * @param values - The values, as produced by toJSON().
   * @returns A new Hyperparameters instance.
   * @throws TypeError if values is not an object, or a bias setting is not a boolean.
   * @throws RangeError if any value fails the constructor's validation.
   */
  static fromJSON(values: Partial<HyperparameterValues>): Hyperparameters {
    if (typeof values !== "object" || values === null) {
      throw new TypeError(`Hyperparameters| Expected an object of hyperparameter values; got ${values}`);
    }
    // The constructor does not validate the booleans, so check them here.
    for (const name of ["qkvBias", "feedForwardBias"] as const) {
      if (values[name] !== undefined && typeof values[name] !== "boolean") {
        throw new TypeError(`Hyperparameters| ${name} must be true or false; got ${values[name]}`);
      }
    }
    return new Hyperparameters(
      values.embeddingSize,
      values.contextLength,
      values.dropoutRate,
      values.mode,
      values.numberAttentionHeads,
      values.numberTransformerBlocks,
      values.batchSize,
      values.qkvBias,
      values.feedForwardBias,
      values.positionEmbeddingStandardDeviation,
      values.tokenEmbeddingStandardDeviation,
      values.linearLayerStandardDeviation,
      values.outputLayerStandardDeviation,
      values.outputLength,
      values.trainingRatio,
      values.numberEpochs,
      values.learningRate,
      values.beta1,
      values.beta2,
      values.optimizerEpsilon,
      values.weightDecay,
      values.numberTrainingExamples,
      values.numberValidationExamples,
      values.temperature,
      values.samplingMethod,
      values.topK,
      values.topP,
      values.seed,
      values.trainingWindowLength
    );
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

  /**
   * Checks that a value is a finite number greater than 0.
   *
   * @param name - The name of the hyperparameter, used in the error message.
   * @param value - The value to check.
   * @throws RangeError if value is 0 or less, infinite or NaN.
   */
  private static validatePositiveNumber(name: string, value: number): void {
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
  private static validateBeta(name: string, value: number): void {
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
  private static validateWeightDecay(weightDecay: number): void {
    if (!(Number.isFinite(weightDecay) && weightDecay >= 0)) {
      throw new RangeError(`Hyperparameters| weightDecay must be a finite number >= 0; got ${weightDecay}`);
    }
  }

  /**
   * Checks that a temperature lies between 0 and 1 inclusive.
   *
   * @param temperature - The temperature to check.
   * @throws RangeError if temperature is less than 0, greater than 1 or NaN.
   */
  private static validateTemperature(temperature: number): void {
    // Written as a negated range check so that NaN, which fails every comparison, is rejected.
    if (!(temperature >= 0 && temperature <= 1)) {
      throw new RangeError(`Hyperparameters| temperature must be between 0 and 1; got ${temperature}`);
    }
  }

  /**
   * Checks that a top-p threshold is greater than 0 and at most 1. At 0 no
   * token would be kept; at 1 every token is.
   *
   * @param topP - The threshold to check.
   * @throws RangeError if topP is not greater than 0 and at most 1, or is NaN.
   */
  private static validateTopP(topP: number): void {
    // Written as a negated range check so that NaN, which fails every comparison, is rejected.
    if (!(topP > 0 && topP <= 1)) {
      throw new RangeError(`Hyperparameters| topP must be greater than 0 and at most 1; got ${topP}`);
    }
  }

  /**
   * Checks that a sampling method is one of the three defined methods.
   *
   * @param samplingMethod - The sampling method to check.
   * @throws RangeError if samplingMethod is not GREEDY_SAMPLING, TOP_K_SAMPLING or TOP_P_SAMPLING.
   */
  private static validateSamplingMethod(samplingMethod: number): void {
    const samplingMethods: number[] = [
      Hyperparameters.GREEDY_SAMPLING,
      Hyperparameters.TOP_K_SAMPLING,
      Hyperparameters.TOP_P_SAMPLING,
    ];
    if (!samplingMethods.includes(samplingMethod)) {
      throw new RangeError(
        `Hyperparameters| samplingMethod must be GREEDY_SAMPLING (${Hyperparameters.GREEDY_SAMPLING}), ` +
        `TOP_K_SAMPLING (${Hyperparameters.TOP_K_SAMPLING}) or ` +
        `TOP_P_SAMPLING (${Hyperparameters.TOP_P_SAMPLING}); got ${samplingMethod}`
      );
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
   * The longest window to train on: the value set, but never more than
   * contextLength, since the position embedding has no rows beyond that. When
   * no value is set, this is contextLength.
   *
   * Training on shorter windows is much faster, and this does not change the
   * shape of the model. Lowering contextLength on a loaded model would instead
   * save a value that no longer matches its position embedding, and the file
   * could then not be loaded again. A short window length also lets a short
   * text, such as The Verdict, train on a model with a long context.
   */
  get trainingWindowLength(): number {
    return this._trainingWindowLength === undefined
      ? this._contextLength
      : Math.min(this._trainingWindowLength, this._contextLength);
  }

  /**
   * Sets the longest window to train on. Set it to undefined to use
   * contextLength again.
   *
   * @throws RangeError if trainingWindowLength is set and is not a positive whole number.
   */
  set trainingWindowLength(trainingWindowLength: number | undefined) {
    if (trainingWindowLength !== undefined) {
      Hyperparameters.validatePositiveInteger("trainingWindowLength", trainingWindowLength);
    }
    this._trainingWindowLength = trainingWindowLength;
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
   * The proportion of each file that TextDataset uses for training; the rest
   * is held back for validation.
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

  /** The size of each AdamW weight update. AdamW reads it on every update, so changing it takes effect at the next batch. */
  get learningRate(): number {
    return this._learningRate;
  }

  /** @throws RangeError if learningRate is not a finite number greater than 0. */
  set learningRate(learningRate: number) {
    Hyperparameters.validatePositiveNumber("learningRate", learningRate);
    this._learningRate = learningRate;
  }

  /** How much of the previous running average of the gradients AdamW keeps at each step (its momentum). */
  get beta1(): number {
    return this._beta1;
  }

  /** @throws RangeError if beta1 is not at least 0 and less than 1. */
  set beta1(beta1: number) {
    Hyperparameters.validateBeta("beta1", beta1);
    this._beta1 = beta1;
  }

  /** How much of the previous running average of the squared gradients AdamW keeps at each step. */
  get beta2(): number {
    return this._beta2;
  }

  /** @throws RangeError if beta2 is not at least 0 and less than 1. */
  set beta2(beta2: number) {
    Hyperparameters.validateBeta("beta2", beta2);
    this._beta2 = beta2;
  }

  /** A small number AdamW adds before dividing, so it never divides by zero. Not LayerNormalization.epsilon. */
  get optimizerEpsilon(): number {
    return this._optimizerEpsilon;
  }

  /** @throws RangeError if optimizerEpsilon is not a finite number greater than 0. */
  set optimizerEpsilon(optimizerEpsilon: number) {
    Hyperparameters.validatePositiveNumber("optimizerEpsilon", optimizerEpsilon);
    this._optimizerEpsilon = optimizerEpsilon;
  }

  /** How strongly AdamW pulls the weights towards 0 at each step. 0 turns weight decay off. */
  get weightDecay(): number {
    return this._weightDecay;
  }

  /** @throws RangeError if weightDecay is negative, infinite or NaN. */
  set weightDecay(weightDecay: number) {
    Hyperparameters.validateWeightDecay(weightDecay);
    this._weightDecay = weightDecay;
  }

  /**
   * The number of examples, such as TinyStories stories, to train on each time
   * a dataset made up of separate examples is loaded. TinyStories averages
   * about 210 tokens per story, so training time grows by roughly 1 hour per
   * 500 stories with smallModel().
   */
  get numberTrainingExamples(): number {
    return this._numberTrainingExamples;
  }

  /** @throws RangeError if numberTrainingExamples is not a positive whole number. */
  set numberTrainingExamples(numberTrainingExamples: number) {
    Hyperparameters.validatePositiveInteger("numberTrainingExamples", numberTrainingExamples);
    this._numberTrainingExamples = numberTrainingExamples;
  }

  /** The number of examples, such as TinyStories stories, to measure the validation loss on. */
  get numberValidationExamples(): number {
    return this._numberValidationExamples;
  }

  /** @throws RangeError if numberValidationExamples is not a positive whole number. */
  set numberValidationExamples(numberValidationExamples: number) {
    Hyperparameters.validatePositiveInteger("numberValidationExamples", numberValidationExamples);
    this._numberValidationExamples = numberValidationExamples;
  }

  /**
   * How random text generation is, between 0 and 1 inclusive. The logits are
   * divided by the temperature before the softmax, so values near 0 make the
   * most likely token ever more certain to be chosen, while 1 leaves the
   * model's probabilities unchanged.
   */
  get temperature(): number {
    return this._temperature;
  }

  /** @throws RangeError if temperature is less than 0, greater than 1 or NaN. */
  set temperature(temperature: number) {
    Hyperparameters.validateTemperature(temperature);
    this._temperature = temperature;
  }

  /**
   * How the next token is chosen during text generation: GREEDY_SAMPLING
   * (always the most likely token), TOP_K_SAMPLING (at random from the k
   * most likely) or TOP_P_SAMPLING (at random from the smallest set of most
   * likely tokens whose probabilities add up to at least p).
   */
  get samplingMethod(): number {
    return this._samplingMethod;
  }

  /** @throws RangeError if samplingMethod is not GREEDY_SAMPLING, TOP_K_SAMPLING or TOP_P_SAMPLING. */
  set samplingMethod(samplingMethod: number) {
    Hyperparameters.validateSamplingMethod(samplingMethod);
    this._samplingMethod = samplingMethod;
  }

  /**
   * The number of most likely tokens TOP_K_SAMPLING chooses between. Values
   * above vocabularySize keep every token.
   */
  get topK(): number {
    return this._topK;
  }

  /** @throws RangeError if topK is not a positive whole number. */
  set topK(topK: number) {
    Hyperparameters.validatePositiveInteger("topK", topK);
    this._topK = topK;
  }

  /**
   * The share of the probability TOP_P_SAMPLING keeps: it chooses between
   * the smallest set of most likely tokens whose probabilities add up to at
   * least topP.
   */
  get topP(): number {
    return this._topP;
  }

  /** @throws RangeError if topP is not greater than 0 and at most 1. */
  set topP(topP: number) {
    Hyperparameters.validateTopP(topP);
    this._topP = topP;
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
