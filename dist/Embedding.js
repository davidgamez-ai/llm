import BPETokenizer from "./BPETokenizer.js";
import Hyperparameters from "./Hyperparameters.js";
import DEBUG from "./Debug.js";
import createSeededRandom from "./SeededRandom.js";
/**
 * Two embedding matrices, each mapping an id (a token id, or a sequence
 * position) to a vector of embeddingSize random values. The vocabulary size
 * and embedding size are taken from Hyperparameters, so they always match
 * the values used elsewhere in the application.
 */
class Embedding {
    /** Number of rows in each embedding matrix, i.e. the number of distinct token ids supported. */
    vocabSize;
    /** Number of columns in each embedding matrix, i.e. the width of each embedding vector. */
    embeddingSize;
    // The token embedding matrix built by build(): one row per vocabulary
    // token, each row embeddingSize values wide. Looked up by token id.
    textEmbeddingMatrix = [];
    // The position embedding matrix built by build(): same dimensions as
    // textEmbeddingMatrix, but looked up by a token's position in the
    // sequence rather than by its id.
    positionEmbeddingMatrix = [];
    /**
     * Creates an Embedding, copying vocabSize and embeddingSize from the given
     * Hyperparameters so this instance always matches the values used
     * elsewhere in the application.
     *
     * @param hyperparameters - Source of vocabularySize and embeddingSize. Defaults to a new Hyperparameters instance.
     */
    constructor(hyperparameters = new Hyperparameters()) {
        this.vocabSize = hyperparameters.vocabularySize;
        this.embeddingSize = hyperparameters.embeddingSize;
    }
    /**
     * Builds two vocabSize x embeddingSize matrices, textEmbeddingMatrix and
     * positionEmbeddingMatrix, each filled with values drawn uniformly from
     * [min, max), stores them on the instance, and logs their dimensions.
     * Both matrices share the same shape and random-generation procedure;
     * they differ only in how getEmbedding looks rows up in them (by token id
     * versus by sequence position).
     *
     * @param seed - Optional PRNG seed. When provided, the same seed always produces the same matrices, since Math.random() cannot be seeded and would otherwise make results unreproducible between runs. When omitted, Math.random() is used and the matrices differ on every call.
     * @param min - Inclusive lower bound of the random range. Defaults to -3.
     * @param max - Exclusive upper bound of the random range. Defaults to 3.
     * @returns void. The resulting matrices are stored on the instance for use by getEmbedding.
     */
    build(seed, min = -3, max = 3) {
        const random = seed === undefined ? Math.random : createSeededRandom(seed);
        const buildMatrix = () => Array.from({ length: this.vocabSize }, () => Array.from({ length: this.embeddingSize }, () => min + random() * (max - min)));
        this.textEmbeddingMatrix = buildMatrix();
        this.positionEmbeddingMatrix = buildMatrix();
        if (DEBUG.EMBEDDING)
            console.log(`Embeddings| Embedding matrices built. Rows: ${this.textEmbeddingMatrix.length}; columns: ${this.textEmbeddingMatrix[0].length}`);
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
     */
    getEmbeddingFromTokens(tokenIds) {
        const tokenEmbeddings = [];
        const positionEmbeddings = [];
        const combinedEmbeddings = [];
        for (let position = 0; position < tokenIds.length; position++) {
            // Look up the current token's row in the text embedding matrix, by id.
            const id = tokenIds[position];
            const tokenEmbedding = this.textEmbeddingMatrix[id];
            tokenEmbeddings.push(tokenEmbedding);
            // Look up this token's row in the position embedding matrix, by its
            // position in the sequence rather than by its id.
            const positionEmbedding = this.positionEmbeddingMatrix[position];
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
        return combinedEmbeddings;
    }
    /**
     * Logs a one or two dimensional array. When DEBUG.VERBOSE is true the
     * whole array is printed with console.table; otherwise only its
     * dimensions are printed, which keeps the output short for large arrays.
     *
     * @param label - Name of the array, printed before its contents or dimensions.
     * @param array - The array to log: a vector, or a matrix with one row per token.
     */
    logArray(label, array) {
        if (DEBUG.VERBOSE) {
            console.log(`Embedding| ${label}:`);
            console.table(array);
        }
        else if (Array.isArray(array[0])) {
            console.log(`Embedding| ${label}. Rows: ${array.length}; columns: ${array[0].length}`);
        }
        else {
            console.log(`Embedding| ${label}. Length: ${array.length}`);
        }
    }
}
export default Embedding;
