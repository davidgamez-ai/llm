import Hyperparameters from "./Hyperparameters.js";
/**
 * Works out how to spend a number of hours of training: how much data to use
 * and how to set the hyperparameters, given the model and the dataset.
 *
 * Training time is proportional to the number of tokens trained on, so:
 *
 *     tokens per hour = PARAMETER_TOKENS_PER_HOUR / number of parameters
 *     tokens          = hours x tokens per hour
 *
 * The sizes of the datasets are measured constants rather than being read from
 * the files, because the TinyStories training file is nearly 2 GB.
 *
 * Pre-training and fine tuning differ in the following ways, following the book:
 *
 * - The learning rate. Pre-training uses the value of the model's size.
 *   Pretrained weights are already nearly right and large updates would damage
 *   them, so fine tuning uses FINE_TUNING_LEARNING_RATE for every size.
 * - The batch size. Fine tuning uses larger batches, for steadier updates.
 * - The most passes over a small text. A pretrained model soon memorizes one.
 */
class TrainingPlanner {
    /**
     * Training speed: tokens trained on per hour multiplied by the number of
     * parameters. Time per token grows in proportion to the number of
     * parameters, so this is nearly the same for every model, and also gives
     * estimates for loaded models of any shape. Measured on a desktop computer
     * (Ryzen 7 3800X) with batches of 2; a faster or slower computer will take
     * proportionally less or more time:
     *
     *     Parameters      Tokens per hour   Multiplied
     *     6.6 million             213,000   1.41 trillion
     *     13.7 million            110,000   1.50 trillion
     *     49.3 million             27,000   1.33 trillion
     *     163 million               8,100   1.32 trillion
     */
    static PARAMETER_TOKENS_PER_HOUR = 1.35e12;
    /** The learning rate for fine tuning, as the book uses for its GPT-2 fine tuning. */
    static FINE_TUNING_LEARNING_RATE = 0.00005;
    /** The batch size for fine tuning, as the book uses for its GPT-2 fine tuning. */
    static FINE_TUNING_BATCH_SIZE = 8;
    /** The batch size for pre-training, as every model size uses by default. */
    static PRETRAINING_BATCH_SIZE = 2;
    /**
     * The longest window to train on, or the model's contextLength if that is
     * shorter: the value the book uses when training. Longer windows mean fewer
     * weight updates from the same tokens, and leave a small text with no
     * validation windows on the large model.
     */
    static WINDOW_LENGTH = 256;
    /** The most passes over a text when pre-training. */
    static MAXIMUM_PRETRAINING_EPOCHS = 50;
    /** The most passes over a text when fine tuning. */
    static MAXIMUM_FINE_TUNING_EPOCHS = 10;
    /**
     * How many times faster the validation loss is measured than the model is
     * trained, per token, since validation only has a forward pass. Measured at
     * 3 to 4 times.
     */
    static VALIDATION_SPEED_RATIO = 3;
    /** The fewest examples to train on, however short the time. */
    static MINIMUM_TRAINING_EXAMPLES = 10;
    /** The fewest batches a run must have, so that there is more than one weight update. */
    static MINIMUM_BATCHES = 2;
    /**
     * How much more than one window a text's validation part is made to hold,
     * since a split by characters does not give an exact number of tokens.
     */
    static VALIDATION_MARGIN = 1.25;
    /** The measured size of The Verdict. */
    static VERDICT = { characters: 20643, trainingTokens: 4763, validationTokens: 547 };
    /** The measured size of Tiny Shakespeare. */
    static SHAKESPEARE = { characters: 1115394, trainingTokens: 301966, validationTokens: 36059 };
    /** The average number of tokens in a TinyStories story. Measured over 2,000 stories. */
    static TOKENS_PER_STORY = 210;
    /**
     * Alpaca's measured examples that fit in a window of each length, and their
     * average number of tokens, shortest window first. The figures for the
     * longest window are used for any window longer than the others.
     */
    static ALPACA_SIZES = [
        { windowLength: 128, examples: 36826, tokensPerExample: 83 },
        { windowLength: 256, examples: 50637, tokensPerExample: 106 },
    ];
    /**
     * Plans a training run.
     *
     * @param kind - Whether the run is pre-training or fine tuning.
     * @param dataset - The dataset to train on.
     * @param hours - The rough number of hours to train for, including validation.
     * @param hyperparameters - The settings of the model: the source of contextLength, embeddingSize and numberValidationExamples.
     * @param parameterCount - The number of parameters in the model.
     * @param parameterTokensPerHour - The training speed. Defaults to PARAMETER_TOKENS_PER_HOUR.
     * @returns The settings for the run.
     */
    static plan(kind, dataset, hours, hyperparameters, parameterCount, parameterTokensPerHour = TrainingPlanner.PARAMETER_TOKENS_PER_HOUR) {
        const tokensPerHour = parameterTokensPerHour / parameterCount;
        const windowLength = Math.min(TrainingPlanner.WINDOW_LENGTH, hyperparameters.contextLength);
        const fineTuning = kind === "fineTune";
        const batchSize = fineTuning ? TrainingPlanner.FINE_TUNING_BATCH_SIZE : TrainingPlanner.PRETRAINING_BATCH_SIZE;
        const learningRate = fineTuning ? TrainingPlanner.FINE_TUNING_LEARNING_RATE : TrainingPlanner.pretrainingLearningRate(hyperparameters.embeddingSize);
        if (dataset === "verdict" || dataset === "shakespeare") {
            const size = dataset === "verdict" ? TrainingPlanner.VERDICT : TrainingPlanner.SHAKESPEARE;
            const maximumEpochs = fineTuning ? TrainingPlanner.MAXIMUM_FINE_TUNING_EPOCHS : TrainingPlanner.MAXIMUM_PRETRAINING_EPOCHS;
            return { learningRate, ...TrainingPlanner.planText(size, hours, tokensPerHour, windowLength, batchSize, maximumEpochs) };
        }
        const alpacaSize = TrainingPlanner.ALPACA_SIZES.find((candidate) => windowLength <= candidate.windowLength)
            ?? TrainingPlanner.ALPACA_SIZES[TrainingPlanner.ALPACA_SIZES.length - 1];
        return {
            learningRate,
            ...(dataset === "alpaca"
                ? TrainingPlanner.planExamples(alpacaSize.tokensPerExample, alpacaSize.examples, hours, tokensPerHour, windowLength, batchSize, hyperparameters.numberValidationExamples)
                : TrainingPlanner.planExamples(TrainingPlanner.TOKENS_PER_STORY, Infinity, hours, tokensPerHour, windowLength, batchSize, hyperparameters.numberValidationExamples)),
        };
    }
    /**
     * Finds the learning rate for pre-training a model, from the model size it is
     * closest to. It is worked out from the width rather than read from the
     * model's hyperparameters, as those may hold a learning rate from fine tuning.
     *
     * @param embeddingSize - The width of the model.
     * @returns The learning rate of the smallest model size at least this wide, or of the largest size.
     */
    static pretrainingLearningRate(embeddingSize) {
        const sizes = [Hyperparameters.smallModel(), Hyperparameters.mediumModel(), Hyperparameters.largeModel()];
        return (sizes.find((size) => embeddingSize <= size.embeddingSize) ?? sizes[sizes.length - 1]).learningRate;
    }
    /**
     * Plans a run on a continuous text. If the time is too short for a whole pass
     * of the text, only its start is used, for one epoch. Otherwise the whole text
     * is repeated for as many epochs as fit, up to maximumEpochs. Validation runs at
     * the end of every epoch, so its time is allowed for in each.
     *
     * @param size - The measured size of the text.
     * @param hours - The rough number of hours to train for.
     * @param tokensPerHour - How many tokens this model trains on in an hour.
     * @param windowLength - The longest window to train on.
     * @param batchSize - The batch size wanted. Reduced if the text cannot fill two batches.
     * @param maximumEpochs - The most passes over the text.
     * @returns The plan, apart from the learning rate.
     */
    static planText(size, hours, tokensPerHour, windowLength, batchSize, maximumEpochs) {
        const validationShare = size.validationTokens / size.trainingTokens;
        const hoursPerTrainingToken = (1 + validationShare / TrainingPlanner.VALIDATION_SPEED_RATIO) / tokensPerHour;
        const affordableTokens = hours / hoursPerTrainingToken;
        // Enough tokens for two batches, and for the validation part to hold one window.
        const fewestTokens = Math.max((TrainingPlanner.MINIMUM_BATCHES * batchSize + 1) * windowLength, Math.ceil((TrainingPlanner.VALIDATION_MARGIN * (windowLength + 1)) / validationShare));
        const partialTokens = Math.max(affordableTokens, fewestTokens);
        const usePartOfText = partialTokens < size.trainingTokens;
        const trainingTokens = usePartOfText ? partialTokens : size.trainingTokens;
        const numberEpochs = usePartOfText ? 1 : Math.min(maximumEpochs, Math.max(1, Math.floor(affordableTokens / size.trainingTokens)));
        const windows = Math.floor(trainingTokens / windowLength);
        return {
            numberEpochs,
            batchSize: Math.min(batchSize, Math.max(1, Math.floor(windows / TrainingPlanner.MINIMUM_BATCHES))),
            trainingWindowLength: windowLength,
            numberTrainingExamples: undefined,
            maximumCharacters: usePartOfText ? Math.ceil((trainingTokens * size.characters) / size.trainingTokens) : undefined,
            estimatedHours: numberEpochs * trainingTokens * hoursPerTrainingToken,
        };
    }
    /**
     * Plans a run on a dataset of separate examples. There is one epoch, and the
     * number of examples sets the time, since these datasets are large enough for
     * every example to be new. The result is never fewer than
     * MINIMUM_TRAINING_EXAMPLES, or than the examples needed for two batches, so a
     * short time can run for longer than asked.
     *
     * @param tokensPerExample - The average number of tokens in an example.
     * @param examplesInDataset - The number of examples in the dataset that can be used.
     * @param hours - The rough number of hours to train for.
     * @param tokensPerHour - How many tokens this model trains on in an hour.
     * @param windowLength - The longest window to train on.
     * @param batchSize - The batch size.
     * @param validationExamples - The number of examples used for validation.
     * @returns The plan, apart from the learning rate.
     */
    static planExamples(tokensPerExample, examplesInDataset, hours, tokensPerHour, windowLength, batchSize, validationExamples) {
        const validationTokens = validationExamples * tokensPerExample;
        const trainingTokens = hours * tokensPerHour - validationTokens / TrainingPlanner.VALIDATION_SPEED_RATIO;
        const examplesForTwoBatches = Math.ceil((TrainingPlanner.MINIMUM_BATCHES * batchSize * windowLength) / tokensPerExample);
        const fewest = Math.max(TrainingPlanner.MINIMUM_TRAINING_EXAMPLES, examplesForTwoBatches);
        const available = examplesInDataset - validationExamples;
        const numberTrainingExamples = Math.min(available, Math.max(fewest, Math.round(trainingTokens / tokensPerExample)));
        return {
            numberEpochs: 1,
            batchSize,
            trainingWindowLength: windowLength,
            numberTrainingExamples,
            maximumCharacters: undefined,
            estimatedHours: (numberTrainingExamples * tokensPerExample + validationTokens / TrainingPlanner.VALIDATION_SPEED_RATIO) / tokensPerHour,
        };
    }
}
export default TrainingPlanner;
