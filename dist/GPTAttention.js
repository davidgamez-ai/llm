import Hyperparameters from "./Hyperparameters.js";
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
    /** Probability, between 0 and 1, that each attention weight is dropped (set to zero) by dropout. */
    dropoutRate;
    /** True in training mode, when dropout is applied to the attention weights; false in inference mode, when it is skipped. */
    training;
    /** The query weight matrix built by build(): embeddingSize rows by weightMatrixColumns columns. */
    queryWeights = [];
    /** The key weight matrix built by build(): embeddingSize rows by weightMatrixColumns columns. */
    keyWeights = [];
    /** The value weight matrix built by build(): embeddingSize rows by weightMatrixColumns columns. */
    valueWeights = [];
    /**
     * Creates a GPTAttention, copying embeddingSize, weightMatrixColumns,
     * dropoutRate and training from the given Hyperparameters so this
     * instance always matches the values used elsewhere in the application,
     * then builds the weight matrices.
     *
     * @param hyperparameters - Source of embeddingSize, weightMatrixColumns, dropoutRate and training. Defaults to a new Hyperparameters instance.
     */
    constructor(hyperparameters = new Hyperparameters()) {
        this.embeddingSize = hyperparameters.embeddingSize;
        this.weightMatrixColumns = hyperparameters.weightMatrixColumns;
        this.dropoutRate = hyperparameters.dropoutRate;
        this.training = hyperparameters.training;
        this.build();
    }
    /**
     * Builds the queryWeights, keyWeights and valueWeights matrices, each
     * embeddingSize rows by weightMatrixColumns columns, and stores them on the
     * instance. Weights are initialized the same way as PyTorch's nn.Linear:
     * drawn uniformly from [-1/sqrt(embeddingSize), 1/sqrt(embeddingSize)).
     *
     * @returns void. The resulting matrices are stored on the instance.
     */
    build() {
        /*
         * PyTorch's nn.Linear uses kaiming_uniform_ with a = sqrt(5), which
         * works out as a uniform distribution over [-bound, bound) with
         * bound = 1 / sqrt(fan_in). Here fan_in is the number of inputs to each
         * output, i.e. embeddingSize. Scaling the range by fan_in keeps the
         * variance of the projected query, key and value vectors roughly
         * independent of the embedding size.
         */
        const bound = 1 / Math.sqrt(this.embeddingSize);
        // A fresh embeddingSize x weightMatrixColumns matrix of random values
        // in [-bound, bound); called once per weight matrix so each gets its own
        // independent set of random values.
        const buildMatrix = () => Array.from({ length: this.embeddingSize }, () => Array.from({ length: this.weightMatrixColumns }, () => (Math.random() * 2 - 1) * bound));
        this.queryWeights = buildMatrix();
        this.keyWeights = buildMatrix();
        this.valueWeights = buildMatrix();
        if (DEBUG.QKV)
            console.log(`GPTAttention| Weight matrices built. Rows: ${this.embeddingSize}; columns: ${this.weightMatrixColumns}`);
        // console.table lays each matrix out as a proper grid of rows and
        // columns, which is far easier to read than the flattened,
        // comma-separated string that logging a nested array directly produces.
        if (DEBUG.QKV) {
            console.log("GPTAttention| Query weights:");
            console.table(this.queryWeights);
            console.log("GPTAttention| Key weights:");
            console.table(this.keyWeights);
            console.log("GPTAttention| Value weights:");
            console.table(this.valueWeights);
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
     * the key dimension and passed through a softmax to give attention weights.
     * A causal mask then zeroes the weights above the diagonal, and each row
     * is re-normalized to sum to 1. In training mode dropout is then applied
     * to give the final attention weights; in inference mode it is skipped.
     * Finally, each token's context vector is the sum of every token's value
     * vector multiplied by the corresponding final attention weight. The
     * intermediate matrices are logged but not returned.
     *
     * @param embeddings - Token embeddings as produced by Embedding.getEmbedding(): one row per token, each row embeddingSize values wide.
     * @returns The context vectors: a tokenCount x weightMatrixColumns matrix where row i is the causally masked, attention-weighted (and, in training mode, dropout-applied) sum of the value vectors of tokens 0..i.
     */
    calculate(embeddings) {
        if (DEBUG.ATTENTION)
            console.log(`GPTAttention| Calculating queries, keys and values for ${embeddings.length} tokens.`);
        // Multiplies a single embedding vector by a weight matrix: each column
        // of the result is the dot product of the vector with that column of
        // the matrix, so a 1 x embeddingSize vector times an embeddingSize x
        // weightMatrixColumns matrix produces a 1 x weightMatrixColumns vector.
        const multiplyVectorByMatrix = (vector, matrix) => Array.from({ length: this.weightMatrixColumns }, (_, column) => vector.reduce((sum, value, row) => sum + value * matrix[row][column], 0));
        const queries = [];
        const keys = [];
        const values = [];
        // For each token in turn, project its embedding vector into its query,
        // key and value vectors using the three weight matrices.
        for (let i = 0; i < embeddings.length; i++) {
            const embedding = embeddings[i];
            const query = multiplyVectorByMatrix(embedding, this.queryWeights);
            const key = multiplyVectorByMatrix(embedding, this.keyWeights);
            const value = multiplyVectorByMatrix(embedding, this.valueWeights);
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
        /*
         * For each token, take the dot product of its query vector with the key
         * vector of every token in the sequence (including itself). The result
         * is a tokenCount x tokenCount matrix in which row i holds the attention
         * scores of token i, i.e. how strongly token i's query matches each
         * token's key.
         */
        const attentionScores = queries.map((query) => keys.map((key) => query.reduce((sum, value, index) => sum + value * key[index], 0)));
        if (DEBUG.ATTENTION) {
            console.log("GPTAttention| Attention scores (row = query token, column = key token):");
            console.table(attentionScores);
        }
        /*
         * Scale each token's attention scores by dividing them by the square
         * root of the key dimension (weightMatrixColumns), then apply a softmax
         * across each row. Scaling stops the dot products growing with the key
         * dimension, which would otherwise push the softmax towards a one-hot
         * output with tiny gradients. The softmax turns each row into attention
         * weights that are all positive and sum to 1. The row's maximum is
         * subtracted before exponentiating to avoid overflow; this does not
         * change the result because softmax is unaffected by adding a constant
         * to every input.
         */
        const scale = Math.sqrt(this.weightMatrixColumns);
        const attentionWeights = attentionScores.map((row) => {
            const scaled = row.map((score) => score / scale);
            const max = Math.max(...scaled);
            const exponentials = scaled.map((score) => Math.exp(score - max));
            const total = exponentials.reduce((sum, value) => sum + value, 0);
            return exponentials.map((value) => value / total);
        });
        if (DEBUG.ATTENTION) {
            console.log("GPTAttention| Attention weights (scaled and softmaxed; each row sums to 1):");
            console.table(attentionWeights);
        }
        /*
         * Build a causal attention mask with the same tokenCount x tokenCount
         * dimensions as the attention weight matrix. Entries on or below the
         * diagonal are 1 and entries above it are 0, so row i only keeps tokens
         * 0..i and hides any token that comes after token i in the sequence.
         */
        const causalMask = attentionWeights.map((row, rowIndex) => row.map((_, columnIndex) => (columnIndex <= rowIndex ? 1 : 0)));
        if (DEBUG.ATTENTION) {
            console.log("GPTAttention| Causal attention mask (1 = visible, 0 = masked):");
            console.table(causalMask);
        }
        /*
         * Multiply the attention weights element by element with the causal
         * mask. Weights on or below the diagonal are kept (multiplied by 1) and
         * weights above the diagonal are zeroed (multiplied by 0), so each token
         * no longer attends to tokens that come after it.
         */
        const maskedAttentionWeights = attentionWeights.map((row, rowIndex) => row.map((weight, columnIndex) => weight * causalMask[rowIndex][columnIndex]));
        if (DEBUG.ATTENTION) {
            console.log("GPTAttention| Masked attention weights (values above the diagonal zeroed):");
            console.table(maskedAttentionWeights);
        }
        /*
         * Re-normalize the masked attention weights by dividing every element
         * in a row by that row's sum, so each row sums to 1 again. Masking
         * removed the weights above the diagonal, leaving rows that sum to less
         * than 1. A row sum can never be zero because the diagonal is always
         * kept and every softmax output is positive.
         */
        const normalizedMaskedAttentionWeights = maskedAttentionWeights.map((row) => {
            const rowSum = row.reduce((sum, weight) => sum + weight, 0);
            return row.map((weight) => weight / rowSum);
        });
        if (DEBUG.ATTENTION) {
            console.log("GPTAttention| Re-normalized masked attention weights (each row sums to 1):");
            console.table(normalizedMaskedAttentionWeights);
        }
        /*
         * Dropout is only applied in training mode. In inference mode the
         * re-normalized masked attention weights are used unchanged as the
         * final attention weights, so the output is deterministic.
         */
        let finalAttentionWeights;
        if (this.training) {
            /*
             * Build a dropout mask with the same tokenCount x tokenCount
             * dimensions as the attention weights. Each entry is independently
             * set to 0 with probability dropoutRate, so on average that
             * proportion of the mask is zero. The remaining entries are set to
             * 1 / (1 - dropoutRate) so that, once the mask is applied, the
             * expected value of each attention weight is unchanged. With the
             * default dropoutRate of 0.5 the kept entries are 2.
             */
            const keepScale = 1 / (1 - this.dropoutRate);
            const dropoutMask = normalizedMaskedAttentionWeights.map((row) => row.map(() => (Math.random() < this.dropoutRate ? 0 : keepScale)));
            if (DEBUG.ATTENTION) {
                console.log(`GPTAttention| Dropout mask (dropout rate ${this.dropoutRate}; kept values = ${keepScale}):`);
                console.table(dropoutMask);
            }
            /*
             * Apply dropout by multiplying the re-normalized masked attention
             * weights element by element with the dropout mask. Dropped weights
             * become 0 and kept weights are scaled by 1 / (1 - dropoutRate).
             */
            finalAttentionWeights = normalizedMaskedAttentionWeights.map((row, rowIndex) => row.map((weight, columnIndex) => weight * dropoutMask[rowIndex][columnIndex]));
            if (DEBUG.ATTENTION)
                console.log("GPTAttention| Final attention weights (after causal mask and dropout):");
        }
        else {
            finalAttentionWeights = normalizedMaskedAttentionWeights;
            console.log("GPTAttention| Inference mode: dropout skipped. Final attention weights (after causal mask):");
        }
        if (DEBUG.ATTENTION)
            console.table(finalAttentionWeights);
        /*
         * For each token, multiply the value vector of every token (including
         * itself) by the corresponding weight from that token's row of the
         * final attention weights, then sum the weighted value vectors. The
         * result is the token's context vector: a blend of the value vectors of
         * that token and the tokens before it, in which the tokens it attends to
         * most contribute most. Later tokens and dropped weights are zero, so
         * they do not contribute. Collecting these gives a tokenCount x
         * weightMatrixColumns matrix.
         */
        const contextVectors = finalAttentionWeights.map((weights) => Array.from({ length: this.weightMatrixColumns }, (_, column) => weights.reduce((sum, weight, token) => sum + weight * values[token][column], 0)));
        if (DEBUG.CONTEXT) {
            console.log("GPTAttention| Context vectors (one row per token):");
            console.table(contextVectors);
        }
        return contextVectors;
    }
}
export default GPTAttention;
