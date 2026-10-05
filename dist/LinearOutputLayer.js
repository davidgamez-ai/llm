import Hyperparameters from "./Hyperparameters.js";
import DEBUG from "./Debug.js";
/**
 * The linear output layer of the GPT model. It holds a weight matrix with
 * one row per vocabulary token, each row embeddingSize values wide, which is
 * used to map each token vector output by the model to a score for every
 * token in the vocabulary. The vocabulary size and embedding size are taken
 * from Hyperparameters, so they always match the values used elsewhere in
 * the application.
 */
class LinearOutputLayer {
    /** Number of rows in the weight matrix, i.e. the number of tokens in the vocabulary. */
    vocabSize;
    /** Number of columns in the weight matrix, i.e. the width of each token vector. */
    embeddingSize;
    /**
     * True in training mode, when logits are calculated for every token; false
     * in inference mode, when they are only calculated for the last token.
     * Read from Hyperparameters on every access, so changing
     * Hyperparameters.training after construction takes effect.
     */
    get training() {
        return this.isTraining();
    }
    /** Reads the Hyperparameters' current training flag. */
    isTraining;
    /** Standard deviation of the normal distribution (mean 0) used to initialize the weight matrix. GPT-2 uses 0.02. */
    standardDeviation;
    /** The weight matrix built by build(): vocabSize rows by embeddingSize columns. */
    weights = [];
    /** Draws from the Hyperparameters' shared random number stream, which is seeded when Hyperparameters.seed is set. */
    random;
    /**
     * Creates a LinearOutputLayer, copying vocabSize, embeddingSize and
     * outputLayerStandardDeviation from the given Hyperparameters so this
     * instance always matches the values used elsewhere in the application.
     *
     * @param hyperparameters - Source of vocabularySize, embeddingSize, outputLayerStandardDeviation, training and the random number stream. Defaults to a new Hyperparameters instance.
     */
    constructor(hyperparameters = new Hyperparameters()) {
        this.vocabSize = hyperparameters.vocabularySize;
        this.embeddingSize = hyperparameters.embeddingSize;
        this.standardDeviation = hyperparameters.outputLayerStandardDeviation;
        this.isTraining = () => hyperparameters.training;
        this.random = () => hyperparameters.random();
    }
    /**
     * Builds the vocabSize x embeddingSize weight matrix and stores it on the
     * instance. Initialized as in GPT-2 (Hugging Face GPT2Model._init_weights),
     * with values drawn from a normal distribution with mean 0 and standard
     * deviation standardDeviation (0.02 by default). Random values come from
     * the Hyperparameters' shared stream, so the matrix is reproducible when
     * Hyperparameters.seed is set.
     *
     * @returns void. The resulting matrix is stored on the instance.
     */
    build() {
        /*
         * Draws one sample from a normal distribution with mean 0 and the given
         * standard deviation, using the Box-Muller transform to turn two uniform
         * samples into a standard normal sample. 1 - random() lies in (0, 1],
         * so Math.log never receives 0.
         */
        const randomNormal = (deviation) => {
            const u1 = 1 - this.random();
            const u2 = this.random();
            return deviation * Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
        };
        /*
         * A small standard deviation keeps the initial logits close together:
         * each logit is a dot product of embeddingSize terms with a normalized
         * vector, so its standard deviation is about
         * standardDeviation * sqrt(embeddingSize), roughly 0.55 for 0.02 and 768.
         * The untrained model therefore gives every token a similar probability,
         * and the initial loss is close to ln(vocabSize), about 10.8.
         */
        this.weights = Array.from({ length: this.vocabSize }, () => Array.from({ length: this.embeddingSize }, () => randomNormal(this.standardDeviation)));
        if (DEBUG.LINEAR_OUTPUT_LAYER)
            console.log(`LinearOutputLayer| Weight matrix built. Rows: ${this.weights.length}; columns: ${this.weights[0].length}`);
    }
    /**
     * Maps each token embedding to a score (logit) for every token in the
     * vocabulary. The score for vocabulary token v is the dot product of the
     * embedding with row v of the weight matrix, so the whole calculation is
     * embeddings (numberOfTokens x embeddingSize) multiplied by the transpose
     * of weights (embeddingSize x vocabSize).
     *
     * In training mode logits are calculated for every token, since the model
     * learns to predict the next token at every position in the sequence. In
     * inference mode only the last token's logits are needed to predict the
     * next token, so the other tokens are skipped, and a softmax is applied to
     * turn the logits into probabilities.
     *
     * @param embeddings - Token embeddings: one row per token, each row embeddingSize values wide.
     * @returns Each row vocabSize values wide: in training mode the logits, one row per token, in token order; in inference mode a single row of softmax probabilities for the last token.
     */
    calculate(embeddings) {
        const selectedEmbeddings = this.training ? embeddings : embeddings.slice(-1);
        const logits = selectedEmbeddings.map((embedding) => this.weights.map((weightRow) => weightRow.reduce((sum, weight, index) => sum + weight * embedding[index], 0)));
        /*
         * In inference mode, apply a softmax to each row of logits to turn them
         * into a probability distribution over the vocabulary: every value
         * positive and the row summing to 1. The row's maximum is subtracted
         * before exponentiating to avoid overflow; this does not change the
         * result because softmax is unaffected by adding a constant to every
         * input. The maximum is found with reduce rather than Math.max(...row),
         * because spreading a vocabulary-sized row as arguments can exceed the
         * engine's argument limit.
         */
        const output = this.training
            ? logits
            : logits.map((row) => {
                const maxLogit = row.reduce((max, value) => Math.max(max, value), -Infinity);
                const exponentiated = row.map((value) => Math.exp(value - maxLogit));
                const total = exponentiated.reduce((sum, value) => sum + value, 0);
                return exponentiated.map((value) => value / total);
            });
        // Print the whole output in verbose mode; otherwise just its dimensions.
        if (DEBUG.LINEAR_OUTPUT_LAYER) {
            console.log("LinearOutputLayer| Inference complete.");
            if (DEBUG.VERBOSE) {
                console.log("LinearOutputLayer| Output (one row per token):");
                console.table(output);
            }
            else {
                console.log(`LinearOutputLayer| Output. Rows: ${output.length}; columns: ${output[0]?.length ?? 0}`);
            }
        }
        return output;
    }
    /**
     * Counts the trainable parameters in the layer: every value in the weight
     * matrix. Returns 0 until build() has been called.
     *
     * @returns The total number of weights currently stored in weights.
     */
    getParameterCount() {
        return this.weights.reduce((sum, row) => sum + row.length, 0);
    }
}
export default LinearOutputLayer;
