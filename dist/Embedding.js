import BPETokenizer from "./BPETokenizer.js";
// An embedding matrix mapping each token id in the vocabulary to a vector
// of embeddingSize random values. The vocabulary size is taken from
// BPETokenizer, so it always matches the tokenizer actually in use.
class Embedding {
    vocabSize;
    embeddingSize;
    // The embedding matrix built by build(): one row per vocabulary token,
    // each row embeddingSize values wide.
    matrix = [];
    constructor(embeddingSize = 3) {
        this.vocabSize = new BPETokenizer().vocabularySize;
        this.embeddingSize = embeddingSize;
    }
    // Builds a vocabSize x embeddingSize matrix of values drawn uniformly
    // from [min, max), stores it on the instance, and logs the first three
    // rows. Passing the same seed always produces the same matrix, since
    // Math.random() cannot be seeded and would otherwise make results
    // unreproducible between runs.
    build(seed, min = -5, max = 5) {
        const random = seed === undefined ? Math.random : this.createSeededRandom(seed);
        this.matrix = Array.from({ length: this.vocabSize }, () => Array.from({ length: this.embeddingSize }, () => min + random() * (max - min)));
        console.log(this.matrix.slice(0, 3));
    }
    // Tokenizes text with BPETokenizer, then looks up each resulting token
    // id's row in the embedding matrix by using the id as an index.
    getEmbedding(text) {
        const tokenIds = new BPETokenizer().encode(text);
        return tokenIds.map((id) => this.matrix[id]);
    }
    // mulberry32: a small, fast seeded PRNG. Deterministically derives a
    // stream of 32-bit states from the seed and scrambles each one via
    // Math.imul/xorshift into a uniform value in [0, 1).
    createSeededRandom(seed) {
        let state = seed;
        return () => {
            state = (state + 0x6d2b79f5) | 0;
            let t = Math.imul(state ^ (state >>> 15), 1 | state);
            t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
            return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
        };
    }
}
export default Embedding;
