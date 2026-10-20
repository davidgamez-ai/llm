import Hyperparameters from "./Hyperparameters.js";
import { randomNormalMatrix } from "./WeightInitialization.js";
import Matrix, { vectorTimesMatrix, matrixTimesVector, addOuterProduct, addToVector, runningMean } from "./Matrix.js";
import Dropout from "./Dropout.js";
import { softmax } from "./Loss.js";
import DEBUG from "./Debug.js";
/**
 * Holds the trainable query, key and value weight matrices used by
 * GPT-style self-attention to project each token embedding into its
 * query, key and value vectors. Each matrix is embeddingSize rows by
 * weightMatrixColumns columns, both taken from Hyperparameters, so they
 * always match the values used elsewhere in the application.
 */
class GPTAttention {
    /** Number of rows in each weight matrix, i.e. the width of each input embedding vector. */
    embeddingSize;
    /** Number of columns in each weight matrix, i.e. the width of each projected query/key/value vector. */
    weightMatrixColumns;
    /** Dropout applied to the attention weights (training mode only). Keeps its mask for backpropagation. */
    attentionDropout;
    /** The query weight matrix built by build(): embeddingSize rows by weightMatrixColumns columns. */
    queryWeights = new Matrix(0, 0);
    /** The key weight matrix built by build(): embeddingSize rows by weightMatrixColumns columns. */
    keyWeights = new Matrix(0, 0);
    /** The value weight matrix built by build(): embeddingSize rows by weightMatrixColumns columns. */
    valueWeights = new Matrix(0, 0);
    /** True if the query, key and value projections add a bias to their outputs; false if they do not. */
    qkvBias;
    /** Query biases built by build(): weightMatrixColumns values. Empty when qkvBias is false. */
    queryBiases = new Float32Array(0);
    /** Key biases built by build(): weightMatrixColumns values. Empty when qkvBias is false. */
    keyBiases = new Float32Array(0);
    /** Value biases built by build(): weightMatrixColumns values. Empty when qkvBias is false. */
    valueBiases = new Float32Array(0);
    /** Standard deviation of the normal distribution (mean 0) used to initialize the weight matrices. GPT-2 uses 0.02. */
    standardDeviation;
    /** Gradients of queryWeights, set by calculateGradients(). Empty until then. */
    queryWeightGradients = new Matrix(0, 0);
    /** Gradients of keyWeights, set by calculateGradients(). Empty until then. */
    keyWeightGradients = new Matrix(0, 0);
    /** Gradients of valueWeights, set by calculateGradients(). Empty until then. */
    valueWeightGradients = new Matrix(0, 0);
    /** Gradients of queryBiases, set by calculateGradients(). Empty until then, and always empty when qkvBias is false. */
    queryBiasGradients = new Float32Array(0);
    /** Gradients of keyBiases, set by calculateGradients(). Empty until then, and always empty when qkvBias is false. */
    keyBiasGradients = new Float32Array(0);
    /** Gradients of valueBiases, set by calculateGradients(). Empty until then, and always empty when qkvBias is false. */
    valueBiasGradients = new Float32Array(0);
    /** The number of windows averaged into the gradients. */
    gradientCount = 0;
    /*
     * Values stored by the most recent update() for backpropagation: the
     * input embeddings, the queries, keys and values, and the attention
     * weights before dropout (needed for the softmax) and after it (needed
     * for the context vectors). One row per token.
     */
    inputs = [];
    queries = [];
    keys = [];
    values = [];
    attentionWeights = [];
    finalAttentionWeights = [];
    /** The errors of the queries, keys and values, stored by backpropagate() for calculating the gradients. One row per token. */
    queryErrors = [];
    keyErrors = [];
    valueErrors = [];
    /** Draws from the Hyperparameters' shared random number stream, which is seeded when Hyperparameters.seed is set. */
    random;
    /**
     * Creates a GPTAttention, copying embeddingSize, weightMatrixColumns,
     * qkvBias and linearLayerStandardDeviation from the given
     * Hyperparameters so this instance always matches the values used
     * elsewhere in the application, creates the attention weights' Dropout,
     * then builds the weight matrices and biases.
     *
     * @param hyperparameters - Source of embeddingSize, weightMatrixColumns, qkvBias, linearLayerStandardDeviation and the random number stream, and passed on to Dropout. Defaults to a new Hyperparameters instance.
     */
    constructor(hyperparameters = new Hyperparameters()) {
        this.embeddingSize = hyperparameters.embeddingSize;
        this.weightMatrixColumns = hyperparameters.weightMatrixColumns;
        this.attentionDropout = new Dropout(hyperparameters);
        this.qkvBias = hyperparameters.qkvBias;
        this.standardDeviation = hyperparameters.linearLayerStandardDeviation;
        this.random = () => hyperparameters.random();
        this.build();
    }
    /**
     * Builds the queryWeights, keyWeights and valueWeights matrices, each
     * embeddingSize rows by weightMatrixColumns columns, and stores them on the
     * instance. Initialized as in GPT-2 (Hugging Face
     * GPT2PreTrainedModel._init_weights, which initializes GPT-2's combined
     * query/key/value layer c_attn this way):
     * - Weights are drawn from a normal distribution with mean 0 and standard
     *   deviation standardDeviation (0.02 by default).
     * - When qkvBias is true, each projection has a bias vector of
     *   weightMatrixColumns values, all 0; otherwise the bias arrays are left
     *   empty.
     *
     * @returns void. The resulting matrices and biases are stored on the instance.
     */
    build() {
        // Called once per weight matrix, so each gets its own independent set of random values.
        this.queryWeights = randomNormalMatrix(this.embeddingSize, this.weightMatrixColumns, this.standardDeviation, this.random);
        this.keyWeights = randomNormalMatrix(this.embeddingSize, this.weightMatrixColumns, this.standardDeviation, this.random);
        this.valueWeights = randomNormalMatrix(this.embeddingSize, this.weightMatrixColumns, this.standardDeviation, this.random);
        // A new Float32Array is filled with zeros, so the biases start at 0.
        const biasLength = this.qkvBias ? this.weightMatrixColumns : 0;
        this.queryBiases = new Float32Array(biasLength);
        this.keyBiases = new Float32Array(biasLength);
        this.valueBiases = new Float32Array(biasLength);
        if (DEBUG.QKV)
            console.log(`GPTAttention| Weight matrices built. Rows: ${this.embeddingSize}; columns: ${this.weightMatrixColumns}`);
        // console.table lays each matrix out as a proper grid of rows and
        // columns, which is far easier to read than the flattened,
        // comma-separated string that logging a nested array directly produces.
        if (DEBUG.QKV) {
            console.log("GPTAttention| Query weights:");
            console.table(this.queryWeights.toArrays());
            console.log("GPTAttention| Key weights:");
            console.table(this.keyWeights.toArrays());
            console.log("GPTAttention| Value weights:");
            console.table(this.valueWeights.toArrays());
        }
    }
    /**
     * For each token's embedding vector, separately multiplies it by
     * queryWeights, keyWeights and valueWeights to project it into that
     * token's query, key and value vectors. Collecting these vectors across
     * every token produces three new matrices: queries, keys and values,
     * each tokenCount rows by weightMatrixColumns columns. Each token's query
     * is then multiplied (dot product) by every token's key to give that
     * token's attention scores. These scores are scaled by the square root of
     * the key dimension, and a causal mask sets the scores above the diagonal
     * (tokens later in the sequence) to -Infinity. A softmax across each row
     * then gives attention weights in which the masked tokens are 0 and each
     * row sums to 1. In training mode dropout is then applied
     * to give the final attention weights; in validation and generation modes
     * it is skipped.
     * Finally, each token's context vector is the sum of every token's value
     * vector multiplied by the corresponding final attention weight. The
     * intermediate matrices are logged but not returned.
     *
     * @param embeddings - Token embeddings as produced by Embedding.getEmbedding(): one row per token, each row embeddingSize values wide.
     * @returns The context vectors: a tokenCount x weightMatrixColumns matrix where row i is the causally masked, attention-weighted (and, in training mode, dropout-applied) sum of the value vectors of tokens 0..i.
     */
    update(embeddings) {
        if (DEBUG.ATTENTION)
            console.log(`GPTAttention| Calculating queries, keys and values for ${embeddings.length} tokens.`);
        const queries = [];
        const keys = [];
        const values = [];
        /*
         * For each token in turn, project its embedding vector into its query,
         * key and value vectors by multiplying it by the three weight matrices
         * and adding the biases (which are empty when qkvBias is false). A
         * 1 x embeddingSize vector times an embeddingSize x weightMatrixColumns
         * matrix produces a 1 x weightMatrixColumns vector.
         */
        for (let i = 0; i < embeddings.length; i++) {
            const embedding = embeddings[i];
            const query = vectorTimesMatrix(embedding, this.queryWeights, this.queryBiases);
            const key = vectorTimesMatrix(embedding, this.keyWeights, this.keyBiases);
            const value = vectorTimesMatrix(embedding, this.valueWeights, this.valueBiases);
            queries.push(query);
            keys.push(key);
            values.push(value);
        }
        // console.table lays each resulting matrix out as a proper grid of
        // rows (tokens) and columns, which is far easier to check by eye than
        // logging a nested array directly, row by row, as the loop went along.
        if (DEBUG.ATTENTION) {
            console.log("GPTAttention| Queries (one row per token):");
            console.table(queries);
            console.log("GPTAttention| Keys (one row per token):");
            console.table(keys);
            console.log("GPTAttention| Values (one row per token):");
            console.table(values);
        }
        const tokenCount = embeddings.length;
        /*
         * For each token, take the dot product of its query vector with the key
         * vector of every token in the sequence (including itself). The result
         * is a tokenCount x tokenCount matrix in which row i holds the attention
         * scores of token i, i.e. how strongly token i's query matches each
         * token's key:
         *
         *     score[i][j] = sum over d of query[i][d] * key[j][d]
         */
        const attentionScores = [];
        for (let i = 0; i < tokenCount; i++) {
            attentionScores.push([]);
            for (let j = 0; j < tokenCount; j++) {
                let dotProduct = 0;
                for (let d = 0; d < this.weightMatrixColumns; d++) {
                    dotProduct += queries[i][d] * keys[j][d];
                }
                attentionScores[i].push(dotProduct);
            }
        }
        if (DEBUG.ATTENTION) {
            console.log("GPTAttention| Attention scores (row = query token, column = key token):");
            console.table(attentionScores);
        }
        /*
         * Scale each token's attention scores by dividing them by the square
         * root of the key dimension (weightMatrixColumns). Scaling stops the dot
         * products growing with the key dimension, which would otherwise push
         * the softmax towards a one-hot output with tiny gradients.
         *
         * At the same time apply the causal mask: the score of every token that
         * comes after token i in the sequence (column > row, above the diagonal)
         * is replaced with -Infinity, so token i can only attend to tokens 0..i.
         * Masking before the softmax, as the book does, means the masked tokens
         * take no part in the softmax at all, so no re-normalization is needed
         * afterwards.
         */
        const scale = Math.sqrt(this.weightMatrixColumns);
        const maskedScores = [];
        for (let i = 0; i < tokenCount; i++) {
            maskedScores.push([]);
            for (let j = 0; j < tokenCount; j++) {
                // Token i may attend to token j only if j is not after i.
                maskedScores[i].push(j <= i ? attentionScores[i][j] / scale : -Infinity);
            }
        }
        if (DEBUG.ATTENTION) {
            console.log("GPTAttention| Scaled and causally masked attention scores (-Infinity = masked):");
            console.table(maskedScores);
        }
        /*
         * Apply a softmax across each row to turn the masked scores into
         * attention weights that are all positive and sum to 1. Because
         * exp(-Infinity) is 0, the masked tokens get a weight of exactly 0.
         *
         * The row's maximum is subtracted before exponentiating to avoid
         * overflow; this does not change the result because softmax is
         * unaffected by adding a constant to every input. The maximum is taken
         * after masking, so it always comes from a visible token. The visible
         * token with that maximum therefore has exp(0) = 1, which means the row
         * total is at least 1 and can never underflow to 0. (Taking the maximum
         * before masking could make every visible exponential underflow to 0 if
         * a future token's score was much larger, giving a NaN row.) The diagonal
         * is never masked, so every row has at least one finite score.
         */
        const attentionWeights = maskedScores.map((row) => softmax(row));
        if (DEBUG.ATTENTION) {
            console.log("GPTAttention| Attention weights (masked softmax; values above the diagonal are 0, each row sums to 1):");
            console.table(attentionWeights);
        }
        /*
         * Dropout on the attention weights, only in training mode. A
         * tokenCount x tokenCount mask is built in which each entry is 0 with
         * probability dropoutRate and 1 / (1 - dropoutRate) otherwise, and the
         * attention weights are multiplied by it element by element, so dropped
         * weights become 0 and the expected value of each weight is unchanged.
         * In validation and generation modes the attention weights are used
         * unchanged as the final attention weights, so the output is
         * deterministic.
         */
        const finalAttentionWeights = this.attentionDropout.update(attentionWeights);
        if (DEBUG.ATTENTION) {
            // The mask is only set when dropout was applied, i.e. in training mode.
            if (this.attentionDropout.mask !== null) {
                console.log(`GPTAttention| Dropout mask (dropout rate ${this.attentionDropout.dropoutRate}):`);
                console.table(this.attentionDropout.mask);
                console.log("GPTAttention| Final attention weights (after causal mask and dropout):");
            }
            else {
                console.log("GPTAttention| Validation or generation mode: dropout skipped. Final attention weights (after causal mask):");
            }
            console.table(finalAttentionWeights);
        }
        /*
         * For each token, multiply the value vector of every token (including
         * itself) by the corresponding weight from that token's row of the
         * final attention weights, then sum the weighted value vectors. The
         * result is the token's context vector: a blend of the value vectors of
         * that token and the tokens before it, in which the tokens it attends to
         * most contribute most. Later tokens and dropped weights are zero, so
         * they do not contribute. Collecting these gives a tokenCount x
         * weightMatrixColumns matrix:
         *
         *     context[i][d] = sum over j of weight[i][j] * value[j][d]
         */
        const contextVectors = [];
        for (let i = 0; i < tokenCount; i++) {
            contextVectors.push([]);
            for (let d = 0; d < this.weightMatrixColumns; d++) {
                let sum = 0;
                for (let j = 0; j < tokenCount; j++) {
                    sum += finalAttentionWeights[i][j] * values[j][d];
                }
                contextVectors[i].push(sum);
            }
        }
        if (DEBUG.CONTEXT) {
            console.log("GPTAttention| Context vectors (one row per token):");
            console.table(contextVectors);
        }
        // Stored for backpropagate() and calculateGradients(). The dropout mask is kept by attentionDropout.
        this.inputs = embeddings;
        this.queries = queries;
        this.keys = keys;
        this.values = values;
        this.attentionWeights = attentionWeights;
        this.finalAttentionWeights = finalAttentionWeights;
        return contextVectors;
    }
    /**
     * Passes the errors at the context vectors back through the attention
     * head to its input embeddings, and stores the errors of the queries,
     * keys and values for calculateGradients(). Each step undoes one step of
     * update(), in reverse order. Unlike the other components, attention mixes
     * tokens: token i's context vector depends on tokens 0..i, so each token's
     * errors collect contributions from every later token that attended to it.
     *
     * @param outputErrors - The errors at the context vectors: one row per token, each weightMatrixColumns values wide.
     * @returns The errors at the input embeddings: one row per token, each embeddingSize values wide.
     * @throws RangeError if outputErrors does not have one row per token given to the most recent update().
     */
    backpropagate(outputErrors) {
        const tokenCount = this.inputs.length;
        if (outputErrors.length !== tokenCount) {
            throw new RangeError(`GPTAttention| Expected errors for ${tokenCount} tokens (from the most recent update()); got ${outputErrors.length}`);
        }
        /*
         * 1. Context vectors. context[i] = sum over j of finalWeight[i][j] * value[j],
         *    so the error of final weight (i, j) is the dot product of context
         *    i's error with value j, and value j's error collects context i's
         *    error times finalWeight[i][j] from every token i that used it.
         */
        const finalWeightErrors = [];
        const valueErrors = this.values.map((value) => value.map(() => 0));
        for (let i = 0; i < tokenCount; i++) {
            finalWeightErrors.push([]);
            for (let j = 0; j < tokenCount; j++) {
                let dotProduct = 0;
                for (let d = 0; d < this.weightMatrixColumns; d++) {
                    dotProduct += outputErrors[i][d] * this.values[j][d];
                    valueErrors[j][d] += this.finalAttentionWeights[i][j] * outputErrors[i][d];
                }
                finalWeightErrors[i].push(dotProduct);
            }
        }
        // 2. Dropout: dropped weights pass no error back; kept ones are scaled as they were in update().
        const weightErrors = this.attentionDropout.backpropagate(finalWeightErrors);
        /*
         * 3. Softmax, row by row. Each weight in a row depends on every score in
         *    that row (they share the same total), which gives
         *
         *        scoreError[i][j] = weight[i][j] * (weightError[i][j] - sum over k of weight[i][k] * weightError[i][k])
         *
         *    The causal mask needs no special handling here: masked positions
         *    have a weight of exactly 0, so their score errors are 0 too.
         *
         * 4. Scaling. Each score was divided by sqrt(weightMatrixColumns), so
         *    its error is divided by the same amount.
         */
        const scale = Math.sqrt(this.weightMatrixColumns);
        const scoreErrors = this.attentionWeights.map((weights, i) => {
            const weightedSum = weights.reduce((sum, weight, k) => sum + weight * weightErrors[i][k], 0);
            return weights.map((weight, j) => (weight * (weightErrors[i][j] - weightedSum)) / scale);
        });
        /*
         * 5. Scores. score[i][j] = query[i] · key[j], so query i's error collects
         *    key j times score (i, j)'s error over every j, and key j's error
         *    collects query i times score (i, j)'s error over every i.
         */
        const queryErrors = this.queries.map((query) => query.map(() => 0));
        const keyErrors = this.keys.map((key) => key.map(() => 0));
        for (let i = 0; i < tokenCount; i++) {
            for (let j = 0; j < tokenCount; j++) {
                for (let d = 0; d < this.weightMatrixColumns; d++) {
                    queryErrors[i][d] += scoreErrors[i][j] * this.keys[j][d];
                    keyErrors[j][d] += scoreErrors[i][j] * this.queries[i][d];
                }
            }
        }
        this.queryErrors = queryErrors;
        this.keyErrors = keyErrors;
        this.valueErrors = valueErrors;
        /*
         * 6. Input embeddings. Each embedding was projected into a query, a key
         *    and a value, so its error is the sum of the errors passed back
         *    through all three projections (errors from every place a value is
         *    used are added together). Each projection is passed back through
         *    its weights transposed: matrixTimesVector, because the weights are
         *    stored input x output.
         */
        return this.inputs.map((_, t) => {
            const fromQuery = matrixTimesVector(this.queryWeights, queryErrors[t]);
            const fromKey = matrixTimesVector(this.keyWeights, keyErrors[t]);
            const fromValue = matrixTimesVector(this.valueWeights, valueErrors[t]);
            return fromQuery.map((error, e) => error + fromKey[e] + fromValue[e]);
        });
    }
    /**
     * Calculates the gradients of the query, key and value weights (and
     * biases, when qkvBias is true) from the input embeddings stored by
     * update() and the errors stored by backpropagate(). Each projection is a
     * linear layer, so its weight gradient is the outer product of the input
     * embedding with the projection's errors, summed over the tokens:
     *
     *     queryWeightGradients = sum over tokens of embedding ⊗ queryError
     *
     * and likewise for the keys and values. Each bias gradient is the sum of
     * the projection's errors. As in dnn, the first window of a batch
     * (batchAverage false) replaces the stored gradients, and each later
     * window (batchAverage true) is folded into their running mean.
     *
     * @param batchAverage - False for the first window of a batch; true to average this window into the stored gradients.
     */
    calculateGradients(batchAverage) {
        const queryWeightGradients = new Matrix(this.embeddingSize, this.weightMatrixColumns);
        const keyWeightGradients = new Matrix(this.embeddingSize, this.weightMatrixColumns);
        const valueWeightGradients = new Matrix(this.embeddingSize, this.weightMatrixColumns);
        const queryBiasGradients = new Float32Array(this.queryBiases.length);
        const keyBiasGradients = new Float32Array(this.keyBiases.length);
        const valueBiasGradients = new Float32Array(this.valueBiases.length);
        for (let token = 0; token < this.inputs.length; token++) {
            addOuterProduct(queryWeightGradients, this.inputs[token], this.queryErrors[token]);
            addOuterProduct(keyWeightGradients, this.inputs[token], this.keyErrors[token]);
            addOuterProduct(valueWeightGradients, this.inputs[token], this.valueErrors[token]);
            if (this.qkvBias) {
                addToVector(queryBiasGradients, this.queryErrors[token]);
                addToVector(keyBiasGradients, this.keyErrors[token]);
                addToVector(valueBiasGradients, this.valueErrors[token]);
            }
        }
        this.gradientCount = batchAverage && this.gradientCount > 0 ? this.gradientCount + 1 : 1;
        this.queryWeightGradients = runningMean(this.queryWeightGradients, queryWeightGradients, this.gradientCount);
        this.keyWeightGradients = runningMean(this.keyWeightGradients, keyWeightGradients, this.gradientCount);
        this.valueWeightGradients = runningMean(this.valueWeightGradients, valueWeightGradients, this.gradientCount);
        this.queryBiasGradients = runningMean(this.queryBiasGradients, queryBiasGradients, this.gradientCount);
        this.keyBiasGradients = runningMean(this.keyBiasGradients, keyBiasGradients, this.gradientCount);
        this.valueBiasGradients = runningMean(this.valueBiasGradients, valueBiasGradients, this.gradientCount);
    }
    /**
     * Updates the query, key and value weights (and biases, when qkvBias is
     * true) from the gradients calculated by calculateGradients(), using the
     * shared optimizer. Weight decay is applied to the weights but not to the
     * biases.
     *
     * @param optimizer - The model's AdamW optimizer. Its step() must already have been called for this batch.
     */
    updateWeights(optimizer) {
        optimizer.updateMatrix(this.queryWeights, this.queryWeightGradients, true);
        optimizer.updateMatrix(this.keyWeights, this.keyWeightGradients, true);
        optimizer.updateMatrix(this.valueWeights, this.valueWeightGradients, true);
        if (this.qkvBias) {
            optimizer.updateVector(this.queryBiases, this.queryBiasGradients, false);
            optimizer.updateVector(this.keyBiases, this.keyBiasGradients, false);
            optimizer.updateVector(this.valueBiases, this.valueBiasGradients, false);
        }
    }
    /**
     * Counts the trainable parameters in this attention head: every value in
     * the query, key and value weight matrices, plus their biases. The biases
     * are empty, and so add nothing, when qkvBias is false.
     *
     * @returns The total number of weights and biases currently stored on the instance.
     */
    getParameterCount() {
        return (this.queryWeights.size + this.keyWeights.size + this.valueWeights.size +
            this.queryBiases.length + this.keyBiases.length + this.valueBiases.length);
    }
}
export default GPTAttention;
