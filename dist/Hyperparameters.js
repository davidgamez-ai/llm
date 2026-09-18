import BPETokenizer from "./BPETokenizer.js";
/**
 * Central store for hyperparameters shared across multiple parts of the
 * application (e.g. Embedding), so that a single instance can be passed
 * around instead of each consumer redeclaring its own copies of the same
 * values.
 */
class Hyperparameters {
    _embeddingSize;
    _contextLength;
    _vocabularySize;
    /**
     * Creates a Hyperparameters instance.
     *
     * @param embeddingSize - The size of each token embedding vector.
     * @param contextLength - The number of tokens of context the model
     * operates over at once.
     */
    constructor(embeddingSize = 3, contextLength = 1024) {
        this._embeddingSize = embeddingSize;
        this._contextLength = contextLength;
        this._vocabularySize = new BPETokenizer().vocabularySize;
    }
    /** The size of each token embedding vector. */
    get embeddingSize() {
        return this._embeddingSize;
    }
    /** The number of tokens of context the model operates over at once. */
    get contextLength() {
        return this._contextLength;
    }
    /**
     * The size of the vocabulary, loaded from BPETokenizer in the
     * constructor so it always matches the tokenizer actually in use.
     */
    get vocabularySize() {
        return this._vocabularySize;
    }
}
export default Hyperparameters;
