import Hyperparameters from "./Hyperparameters.js";
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
    /** The query weight matrix built by build(): embeddingSize rows by weightMatrixColumns columns. */
    queryWeights = [];
    /** The key weight matrix built by build(): embeddingSize rows by weightMatrixColumns columns. */
    keyWeights = [];
    /** The value weight matrix built by build(): embeddingSize rows by weightMatrixColumns columns. */
    valueWeights = [];
    /**
     * Creates a GPTAttention, copying embeddingSize and weightMatrixColumns
     * from the given Hyperparameters so this instance always matches the
     * values used elsewhere in the application, then builds the weight
     * matrices.
     *
     * @param hyperparameters - Source of embeddingSize and weightMatrixColumns. Defaults to a new Hyperparameters instance.
     */
    constructor(hyperparameters = new Hyperparameters()) {
        this.embeddingSize = hyperparameters.embeddingSize;
        this.weightMatrixColumns = hyperparameters.weightMatrixColumns;
        this.build();
    }
    /**
     * Builds the queryWeights, keyWeights and valueWeights matrices, each
     * embeddingSize rows by weightMatrixColumns columns, filled with random
     * values drawn uniformly from [0, 1), and stores them on the instance.
     *
     * @returns void. The resulting matrices are stored on the instance.
     */
    build() {
        // A fresh embeddingSize x weightMatrixColumns matrix of random values
        // in [0, 1); called once per weight matrix so each gets its own
        // independent set of random values.
        const buildMatrix = () => Array.from({ length: this.embeddingSize }, () => Array.from({ length: this.weightMatrixColumns }, () => Math.random()));
        this.queryWeights = buildMatrix();
        this.keyWeights = buildMatrix();
        this.valueWeights = buildMatrix();
        console.log(`GPTAttention| Weight matrices built. Rows: ${this.embeddingSize}; columns: ${this.weightMatrixColumns}`);
        // console.table lays each matrix out as a proper grid of rows and
        // columns, which is far easier to read than the flattened,
        // comma-separated string that logging a nested array directly produces.
        console.log("GPTAttention| Query weights:");
        console.table(this.queryWeights);
        console.log("GPTAttention| Key weights:");
        console.table(this.keyWeights);
        console.log("GPTAttention| Value weights:");
        console.table(this.valueWeights);
    }
    /**
     * For each token's embedding vector, separately multiplies it by
     * queryWeights, keyWeights and valueWeights to project it into that
     * token's query, key and value vectors. Collecting these vectors across
     * every token produces three new matrices: queries, keys and values,
     * each tokenCount rows by weightMatrixColumns columns.
     *
     * @param embeddings - Token embeddings as produced by Embedding.getEmbedding(): one row per token, each row embeddingSize values wide.
     * @returns An object with queries, keys and values: one projected vector per token, in token order, for each of the three weight matrices.
     */
    calculate(embeddings) {
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
        console.log("GPTAttention| Queries (one row per token):");
        console.table(queries);
        console.log("GPTAttention| Keys (one row per token):");
        console.table(keys);
        console.log("GPTAttention| Values (one row per token):");
        console.table(values);
        return { queries, keys, values };
    }
}
export default GPTAttention;
