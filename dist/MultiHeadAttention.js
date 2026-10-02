import Hyperparameters from "./Hyperparameters.js";
import GPTAttention from "./GPTAttention.js";
import DEBUG from "./Debug.js";
/**
 * Runs several GPTAttention heads over the same token embeddings,
 * concatenates their context vectors, then passes the result through a
 * trainable linear output projection. Each head has its own independently
 * initialized query, key and value weight matrices, so each can learn to
 * attend to different relationships between tokens; the output projection
 * learns how to combine the heads' outputs. The number of heads is taken
 * from Hyperparameters.numberAttentionHeads.
 */
class MultiHeadAttention {
    /** The attention heads, one GPTAttention instance per head, in head order. */
    attentionHeads;
    /** Width of each token embedding; the concatenated context vectors should be the same width. */
    embeddingSize;
    /** Number of transformer blocks in the model, used to scale the output projection's initial weights. */
    numberTransformerBlocks;
    /** Output projection weights built by build(): embeddingSize rows by embeddingSize columns. */
    outputProjectionWeights = [];
    /** Output projection biases built by build(): embeddingSize values. */
    outputProjectionBiases = [];
    /** Draws from the Hyperparameters' shared random number stream, which is seeded when Hyperparameters.seed is set. */
    random;
    /**
     * Creates a MultiHeadAttention with numberAttentionHeads GPTAttention
     * heads, each built from the same Hyperparameters so they all share the
     * same dimensions, dropout rate and training mode, then builds the
     * output projection.
     *
     * @param hyperparameters - Source of numberAttentionHeads, embeddingSize, numberTransformerBlocks and the random number stream, and passed on to each GPTAttention head. Defaults to a new Hyperparameters instance.
     */
    constructor(hyperparameters = new Hyperparameters()) {
        this.embeddingSize = hyperparameters.embeddingSize;
        this.numberTransformerBlocks = hyperparameters.numberTransformerBlocks;
        this.random = () => hyperparameters.random();
        this.attentionHeads = Array.from({ length: hyperparameters.numberAttentionHeads }, () => new GPTAttention(hyperparameters));
        if (DEBUG.ATTENTION)
            console.log(`MultiHeadAttention| Created ${this.attentionHeads.length} attention heads.`);
        this.build();
    }
    /**
     * Builds the output projection: an embeddingSize x embeddingSize weight
     * matrix and an embeddingSize bias vector, stored on the instance.
     * Initialized as in GPT-2 (Hugging Face GPT2Model._init_weights):
     * - Weights are drawn from a normal distribution with mean 0 and standard
     *   deviation 0.02 / sqrt(2 * numberTransformerBlocks).
     * - Biases are set to 0.
     *
     * @returns void. The resulting weights and biases are stored on the instance.
     */
    build() {
        /*
         * GPT-2's base standard deviation for linear layers is 0.02. The output
         * projection writes into the residual stream, and every transformer
         * block adds two such contributions to it (attention and feed forward),
         * so the GPT-2 paper scales these weights by 1 / sqrt(2 * numberOfBlocks)
         * to stop the residual stream's variance growing with the model's depth.
         */
        const standardDeviation = 0.02 / Math.sqrt(2 * this.numberTransformerBlocks);
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
        this.outputProjectionWeights = Array.from({ length: this.embeddingSize }, () => Array.from({ length: this.embeddingSize }, () => randomNormal(standardDeviation)));
        this.outputProjectionBiases = new Array(this.embeddingSize).fill(0);
        if (DEBUG.ATTENTION)
            console.log(`MultiHeadAttention| Output projection built. Rows: ${this.embeddingSize}; columns: ${this.embeddingSize}; ` +
                `weight standard deviation: ${standardDeviation}`);
    }
    /**
     * Calls calculate() on each attention head in turn with the same
     * embeddings, then concatenates the heads' context vectors token by
     * token. Row i of the concatenation is head 0's context vector for token
     * i, followed by head 1's, and so on, giving a tokenCount x
     * (numberAttentionHeads * weightMatrixColumns) matrix. Because
     * weightMatrixColumns = embeddingSize / numberAttentionHeads, each row
     * should be embeddingSize wide; an error is logged for any row that is not.
     * Each concatenated row is then passed through the output projection.
     *
     * @param embeddings - Token embeddings as produced by Embedding.getEmbedding(): one row per token, each row embeddingSize values wide.
     * @returns The projected context vectors: one row per token, in token order, each embeddingSize values wide.
     */
    calculate(embeddings) {
        // One tokenCount x weightMatrixColumns matrix of context vectors per head.
        const headContextVectors = this.attentionHeads.map((head, headIndex) => {
            if (DEBUG.ATTENTION)
                console.log(`MultiHeadAttention| Calculating attention head ${headIndex}.`);
            return head.calculate(embeddings);
        });
        /*
         * Concatenate along the last dimension: for each token, join the
         * context vectors that every head produced for that token into a
         * single row, in head order.
         */
        const contextVectors = embeddings.map((_, token) => headContextVectors.flatMap((headVectors) => headVectors[token]));
        /*
         * Check that every concatenated context vector is embeddingSize wide,
         * so the output can be fed to later layers that expect token vectors
         * of the same width as the embeddings. Mismatches are always logged;
         * success is only logged when attention debugging is on.
         */
        const wrongLengthTokens = contextVectors
            .map((vector, token) => ({ token, length: vector.length }))
            .filter(({ length }) => length !== this.embeddingSize);
        if (wrongLengthTokens.length > 0) {
            wrongLengthTokens.forEach(({ token, length }) => console.error(`MultiHeadAttention| Context vector for token ${token} has length ${length}; expected embeddingSize ${this.embeddingSize}.`));
        }
        else if (DEBUG.ATTENTION) {
            console.log(`MultiHeadAttention| All context vectors have length embeddingSize (${this.embeddingSize}).`);
        }
        if (DEBUG.CONTEXT) {
            console.log("MultiHeadAttention| Concatenated context vectors (one row per token):");
            console.table(contextVectors);
        }
        /*
         * Output projection: each output value is the dot product of the
         * concatenated context vector with one column of
         * outputProjectionWeights, plus that column's bias.
         */
        const projectedVectors = contextVectors.map((vector) => this.outputProjectionBiases.map((bias, column) => vector.reduce((sum, value, row) => sum + value * this.outputProjectionWeights[row][column], bias)));
        if (DEBUG.CONTEXT) {
            console.log("MultiHeadAttention| Projected context vectors (one row per token):");
            console.table(projectedVectors);
        }
        return projectedVectors;
    }
    /**
     * Counts the trainable parameters: those of every attention head plus the
     * output projection's weights and biases. The concatenation itself has no
     * parameters.
     *
     * @returns The total number of parameters in every GPTAttention head and the output projection.
     */
    getParameterCount() {
        const headParameters = this.attentionHeads.reduce((sum, head) => sum + head.getParameterCount(), 0);
        const projectionWeights = this.outputProjectionWeights.reduce((sum, row) => sum + row.length, 0);
        return headParameters + projectionWeights + this.outputProjectionBiases.length;
    }
}
export default MultiHeadAttention;
