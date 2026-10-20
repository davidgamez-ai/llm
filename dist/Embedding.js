import BPETokenizer from "./BPETokenizer.js";
import Hyperparameters from "./Hyperparameters.js";
import { randomNormalMatrix } from "./WeightInitialization.js";
import Matrix, { addToVector, runningMean } from "./Matrix.js";
import DEBUG from "./Debug.js";
/**
 * Two embedding matrices, each mapping an id (a token id, or a sequence
 * position) to a vector of embeddingSize random values. The token embedding
 * matrix has one row per vocabulary token; the position embedding matrix has
 * one row per position in the context window, as in GPT-2. The vocabulary
 * size, context length and embedding size are taken from Hyperparameters, so
 * they always match the values used elsewhere in the application.
 */
class Embedding {
    /** Number of rows in the token embedding matrix, i.e. the number of distinct token ids supported. */
    vocabSize;
    /** Number of rows in the position embedding matrix, i.e. the maximum number of tokens in a sequence. */
    contextLength;
    /** Number of columns in each embedding matrix, i.e. the width of each embedding vector. */
    embeddingSize;
    /**
     * Standard deviation of the normal distribution (mean 0) used to
     * initialize the position embedding matrix. OpenAI's original GPT-2
     * TensorFlow code uses 0.01 for its position embeddings (wpe); Hugging
     * Face's implementation uses 0.02.
     */
    positionEmbeddingStandardDeviation;
    /**
     * Standard deviation of the normal distribution (mean 0) used to
     * initialize the token embedding matrix. GPT-2 uses 0.02 for its token
     * embeddings (wte).
     */
    tokenEmbeddingStandardDeviation;
    /**
     * The token embedding matrix built by build(): one row per vocabulary
     * token, each row embeddingSize values wide. Looked up by token id.
     * Read-only: the matrix itself cannot be replaced from outside, but its
     * values can be changed (e.g. by the optimizer or a gradient check).
     */
    get textEmbeddingMatrix() {
        return this._textEmbeddingMatrix;
    }
    /**
     * The position embedding matrix built by build(): one row per position in
     * the context window (contextLength rows), each row embeddingSize values
     * wide. Looked up by a token's position in the sequence rather than by its
     * id. Read-only in the same way as textEmbeddingMatrix.
     */
    get positionEmbeddingMatrix() {
        return this._positionEmbeddingMatrix;
    }
    /**
     * Gradients of textEmbeddingMatrix, the same shape, set by
     * calculateGradients(). Only the rows of tokens that appeared in the
     * window are non-zero. Empty until calculateGradients() is first called.
     */
    textEmbeddingGradients = new Matrix(0, 0);
    /** Gradients of positionEmbeddingMatrix, the same shape, set by calculateGradients(). Empty until then. */
    positionEmbeddingGradients = new Matrix(0, 0);
    /** The number of windows averaged into the gradients. */
    gradientCount = 0;
    /** Backing field for textEmbeddingMatrix. */
    _textEmbeddingMatrix = new Matrix(0, 0);
    /** Backing field for positionEmbeddingMatrix. */
    _positionEmbeddingMatrix = new Matrix(0, 0);
    /** The token ids given to the most recent getEmbeddingFromTokens(), stored for calculating the gradients. */
    tokenIds = [];
    /** The errors given to the most recent backpropagate(), stored for calculating the gradients. */
    outputErrors = [];
    /** Draws from the Hyperparameters' shared random number stream, which is seeded when Hyperparameters.seed is set. */
    random;
    /**
     * Creates an Embedding, copying vocabSize, contextLength, embeddingSize,
     * positionEmbeddingStandardDeviation and tokenEmbeddingStandardDeviation
     * from the given Hyperparameters so this instance always matches the
     * values used elsewhere in the application.
     *
     * @param hyperparameters - Source of vocabularySize, contextLength, embeddingSize, positionEmbeddingStandardDeviation, tokenEmbeddingStandardDeviation and the random number stream. Defaults to a new Hyperparameters instance.
     */
    constructor(hyperparameters = new Hyperparameters()) {
        this.vocabSize = hyperparameters.vocabularySize;
        this.contextLength = hyperparameters.contextLength;
        this.embeddingSize = hyperparameters.embeddingSize;
        this.positionEmbeddingStandardDeviation = hyperparameters.positionEmbeddingStandardDeviation;
        this.tokenEmbeddingStandardDeviation = hyperparameters.tokenEmbeddingStandardDeviation;
        this.random = () => hyperparameters.random();
    }
    /**
     * Builds the two embedding matrices, stores them on the instance, and logs
     * their dimensions:
     * - textEmbeddingMatrix: vocabSize x embeddingSize, initialized as in
     *   GPT-2 (Hugging Face GPT2PreTrainedModel._init_weights), with values drawn from a
     *   normal distribution with mean 0 and standard deviation
     *   tokenEmbeddingStandardDeviation (0.02 by default).
     * - positionEmbeddingMatrix: contextLength x embeddingSize, initialized as
     *   in OpenAI's original GPT-2 TensorFlow code (model.py, wpe), with values
     *   drawn from a normal distribution with mean 0 and standard deviation
     *   positionEmbeddingStandardDeviation (0.01 by default). Hugging Face's
     *   GPT2PreTrainedModel._init_weights uses 0.02 for the position embeddings instead,
     *   the same as for the token embeddings.
     *
     * Random values come from the Hyperparameters' shared stream, so the
     * matrices are reproducible when Hyperparameters.seed is set.
     *
     * @returns void. The resulting matrices are stored on the instance for use by getEmbedding.
     */
    build() {
        this._textEmbeddingMatrix = randomNormalMatrix(this.vocabSize, this.embeddingSize, this.tokenEmbeddingStandardDeviation, this.random);
        this._positionEmbeddingMatrix = randomNormalMatrix(this.contextLength, this.embeddingSize, this.positionEmbeddingStandardDeviation, this.random);
        if (DEBUG.EMBEDDING) {
            console.log(`Embeddings| Token embedding matrix built. Rows: ${this._textEmbeddingMatrix.rows}; columns: ${this._textEmbeddingMatrix.columns}`);
            console.log(`Embeddings| Position embedding matrix built. Rows: ${this._positionEmbeddingMatrix.rows}; columns: ${this._positionEmbeddingMatrix.columns}`);
        }
    }
    /**
     * Tokenizes text with BPETokenizer and returns the combined embeddings of
     * the resulting tokens, as calculated by getEmbeddingFromTokens.
     *
     * @param text - Raw input text to embed.
     * @returns One combined embedding vector per token, in token order. Each vector is the token's textEmbeddingMatrix row plus its positionEmbeddingMatrix row.
     */
    getEmbedding(text) {
        return this.getEmbeddingFromTokens(new BPETokenizer().encode(text));
    }
    /**
     * For each token id, looks up its row in textEmbeddingMatrix by token id,
     * and its row in positionEmbeddingMatrix by the token's position in the
     * sequence, and adds the two rows together. Each stage (token ids, token
     * embeddings, position embeddings, combined embeddings) is stored in its
     * own array and logged so intermediate results can be inspected, rather
     * than only returning the final combined result.
     *
     * @param tokenIds - Token ids of already tokenized text, in sequence order.
     * @returns One combined embedding vector per token, in token order. Each vector is the token's textEmbeddingMatrix row plus its positionEmbeddingMatrix row.
     * @throws Error if there are more tokens than contextLength, since the position embedding matrix has no row for positions beyond the context window.
     * @throws RangeError if any token id is not a whole number from 0 to vocabSize - 1, since the token embedding matrix has no row for it.
     */
    getEmbeddingFromTokens(tokenIds) {
        if (tokenIds.length > this.contextLength) {
            throw new Error(`Embedding| Sequence of ${tokenIds.length} tokens exceeds the context length of ${this.contextLength}`);
        }
        /*
         * Check every id before looking any up. An id outside the vocabulary
         * (or a fraction or NaN) has no row in textEmbeddingMatrix, so the
         * lookup would return undefined and fail later with a confusing
         * TypeError, far from the cause.
         */
        for (let position = 0; position < tokenIds.length; position++) {
            const id = tokenIds[position];
            if (!(Number.isInteger(id) && id >= 0 && id < this.vocabSize)) {
                throw new RangeError(`Embedding| Token id ${id} at position ${position} is outside the vocabulary (0 to ${this.vocabSize - 1})`);
            }
        }
        // The token and position embedding rows are views onto the matrices, not copies.
        const tokenEmbeddings = [];
        const positionEmbeddings = [];
        const combinedEmbeddings = [];
        for (let position = 0; position < tokenIds.length; position++) {
            // Look up the current token's row in the text embedding matrix, by id.
            const id = tokenIds[position];
            const tokenEmbedding = this._textEmbeddingMatrix.getRow(id);
            tokenEmbeddings.push(tokenEmbedding);
            // Look up this token's row in the position embedding matrix, by its
            // position in the sequence rather than by its id.
            const positionEmbedding = this._positionEmbeddingMatrix.getRow(position);
            positionEmbeddings.push(positionEmbedding);
            // Combine the token embedding and position embedding element-wise so
            // the returned vector encodes both the token's identity and its
            // position in the sequence.
            const combinedEmbedding = [];
            for (let i = 0; i < this.embeddingSize; i++) {
                combinedEmbedding.push(tokenEmbedding[i] + positionEmbedding[i]);
            }
            combinedEmbeddings.push(combinedEmbedding);
        }
        // Log every stage, not just the final result, so each step of the
        // calculation can be checked independently.
        if (DEBUG.EMBEDDING) {
            this.logArray("Token ids", tokenIds);
            this.logArray("Token embeddings", tokenEmbeddings);
            this.logArray("Position embeddings", positionEmbeddings);
            this.logArray("Combined embeddings", combinedEmbeddings);
        }
        // Stored for calculateGradients(): the gradients go to the rows that were looked up.
        this.tokenIds = tokenIds;
        return combinedEmbeddings;
    }
    /**
     * Stores the errors at the combined embeddings for calculateGradients().
     * This is where backpropagation ends. The inputs to the embedding are
     * token ids, which are whole numbers chosen from the vocabulary rather
     * than values that can be nudged up or down, so there are no errors to
     * pass back: nothing comes before the embedding that could be trained.
     *
     * @param outputErrors - The errors at the combined embeddings: one row per token, each embeddingSize values wide.
     * @throws RangeError if outputErrors does not have one row per token given to the most recent getEmbeddingFromTokens().
     */
    backpropagate(outputErrors) {
        if (outputErrors.length !== this.tokenIds.length) {
            throw new RangeError(`Embedding| Expected errors for ${this.tokenIds.length} tokens (from the most recent getEmbeddingFromTokens()); got ${outputErrors.length}`);
        }
        this.outputErrors = outputErrors;
    }
    /**
     * Calculates the gradients of both embedding matrices from the token ids
     * stored by getEmbeddingFromTokens() and the errors stored by
     * backpropagate(). Each combined embedding is the token's row plus the
     * position's row, added together, so each row's gradient is simply the
     * error of the combined embedding it was added into:
     *
     *     textEmbeddingGradients[tokenId[t]] += outputError[t]
     *     positionEmbeddingGradients[t]     += outputError[t]
     *
     * Only the rows of tokens that appeared in the window receive a gradient;
     * every other row's gradient is 0. A token that appears more than once
     * receives the sum of the errors from every position it appears in, since
     * the same row was used each time.
     *
     * The token gradient matrix is stored in full, which is simple but wasteful
     * (it is as large as the vocabulary); real implementations store only the
     * rows that were used.
     *
     * As in dnn, the first window of a batch (batchAverage false) replaces the
     * stored gradients, and each later window (batchAverage true) is folded
     * into their running mean.
     *
     * @param batchAverage - False for the first window of a batch; true to average this window into the stored gradients.
     */
    calculateGradients(batchAverage) {
        const textEmbeddingGradients = new Matrix(this.vocabSize, this.embeddingSize);
        const positionEmbeddingGradients = new Matrix(this.contextLength, this.embeddingSize);
        for (let position = 0; position < this.outputErrors.length; position++) {
            // getRow returns a view onto the gradient matrix, so adding to it changes the matrix.
            addToVector(textEmbeddingGradients.getRow(this.tokenIds[position]), this.outputErrors[position]);
            addToVector(positionEmbeddingGradients.getRow(position), this.outputErrors[position]);
        }
        this.gradientCount = batchAverage && this.gradientCount > 0 ? this.gradientCount + 1 : 1;
        this.textEmbeddingGradients = runningMean(this.textEmbeddingGradients, textEmbeddingGradients, this.gradientCount);
        this.positionEmbeddingGradients = runningMean(this.positionEmbeddingGradients, positionEmbeddingGradients, this.gradientCount);
    }
    /**
     * Counts the trainable parameters in the embedding: every value in the
     * token embedding matrix and the position embedding matrix. Returns 0
     * until build() has been called.
     *
     * @returns The total number of values currently stored in textEmbeddingMatrix and positionEmbeddingMatrix.
     */
    getParameterCount() {
        return this._textEmbeddingMatrix.size + this._positionEmbeddingMatrix.size;
    }
    /**
     * Logs a one or two dimensional array. When DEBUG.VERBOSE is true the
     * whole array is printed with console.table; otherwise only its
     * dimensions are printed, which keeps the output short for large arrays.
     *
     * @param label - Name of the array, printed before its contents or dimensions.
     * @param array - The array to log: a vector, or one row per token. Rows may be Float32Arrays (rows of a Matrix).
     */
    logArray(label, array) {
        const isTwoDimensional = typeof array[0] === "object";
        if (DEBUG.VERBOSE) {
            console.log(`Embedding| ${label}:`);
            // Convert Float32Array rows to plain arrays so console.table lays them out as a grid.
            console.table(isTwoDimensional ? array.map((row) => Array.from(row)) : array);
        }
        else if (isTwoDimensional) {
            console.log(`Embedding| ${label}. Rows: ${array.length}; columns: ${array[0].length}`);
        }
        else {
            console.log(`Embedding| ${label}. Length: ${array.length}`);
        }
    }
}
export default Embedding;
