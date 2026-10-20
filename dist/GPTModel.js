import Hyperparameters from "./Hyperparameters.js";
import Embedding from "./Embedding.js";
import Transformer from "./Transformer.js";
import LayerNormalization from "./LayerNormalization.js";
import LinearOutputLayer from "./LinearOutputLayer.js";
import Dropout from "./Dropout.js";
import DEBUG from "./Debug.js";
/**
 * The GPT model. It converts a sequence of token ids into embeddings,
 * applies dropout to them (training mode only), then passes them through a
 * stack of transformer blocks in sequence, with the output of each block
 * becoming the input to the next. Layer normalization is then applied to the
 * output of the last transformer block, and the linear output layer turns
 * the result into logits: for every token in training and validation modes,
 * or for the last token only in generation mode.
 */
class GPTModel {
    /** Converts token ids into combined token and position embeddings. */
    embedding;
    /** Dropout applied to the embeddings (training mode only). Keeps its mask for backpropagation. */
    embeddingDropout;
    /** The transformer blocks, applied in array order. There are numberTransformerBlocks of them. */
    transformers;
    /** Layer normalization applied to the output of the last transformer block, before the linear output layer. */
    finalLayerNormalization;
    /** Maps the final normalized token vectors to a score for every token in the vocabulary. */
    linearOutputLayer;
    /**
     * Creates a GPTModel with an Embedding and a LinearOutputLayer, whose
     * matrices are built immediately, a Dropout for the embeddings,
     * numberTransformerBlocks Transformer instances and a final
     * LayerNormalization, all built from the same Hyperparameters so their
     * dimensions match. Every component draws its random values from the
     * Hyperparameters' shared stream, so the whole model is reproducible when
     * Hyperparameters.seed is set.
     *
     * @param hyperparameters - Source of numberTransformerBlocks, and passed on to Embedding, Dropout, each Transformer, LayerNormalization and LinearOutputLayer. Defaults to a new Hyperparameters instance.
     */
    constructor(hyperparameters = new Hyperparameters()) {
        this.embedding = new Embedding(hyperparameters);
        this.embedding.build();
        this.embeddingDropout = new Dropout(hyperparameters);
        this.transformers = Array.from({ length: hyperparameters.numberTransformerBlocks }, () => new Transformer(hyperparameters));
        this.finalLayerNormalization = new LayerNormalization(hyperparameters);
        /*
         * The output layer's weight matrix has the same dimensions as the
         * embedding's token matrix and is built the same way. Both draw from
         * the same shared stream, at different points in it, so the two
         * matrices are independent rather than identical.
         */
        this.linearOutputLayer = new LinearOutputLayer(hyperparameters);
        this.linearOutputLayer.build();
    }
    /**
     * Runs tokenized text through the model: embedding, dropout on the
     * embeddings (training mode only), each transformer block in turn, layer
     * normalization of the final block's output, then the linear output layer.
     *
     * @param tokenIds - Token ids of the tokenized input text, in sequence order.
     * @returns The logits, each row vocabularySize values wide: in training and validation modes one row per token, in token order; in generation mode a single row for the last token, scoring every candidate next token.
     */
    update(tokenIds) {
        const embeddings = this.embedding.getEmbeddingFromTokens(tokenIds);
        // Dropout on the embeddings, only in training mode; in other modes the embeddings are used unchanged.
        const droppedEmbeddings = this.embeddingDropout.update(embeddings);
        // The mask is only set when dropout was applied, i.e. in training mode.
        if (DEBUG.GPT_MODEL && this.embeddingDropout.mask !== null) {
            console.log(`GPTModel| Embeddings after dropout (dropout rate ${this.embeddingDropout.dropoutRate}):`);
            if (DEBUG.VERBOSE)
                console.table(droppedEmbeddings);
        }
        // Pass the embeddings through each transformer block in sequence, feeding each block's output into the next.
        let transformerOutput = droppedEmbeddings;
        for (const transformer of this.transformers) {
            transformerOutput = transformer.update(transformerOutput);
        }
        // Final layer normalization, applied to each token vector independently, as in GPT-2.
        const normalizedOutput = this.finalLayerNormalization.update(transformerOutput);
        if (DEBUG.GPT_MODEL) {
            console.log("GPTModel| Output of final layer normalization (one row per token):");
            if (DEBUG.VERBOSE)
                console.table(normalizedOutput);
        }
        // Logits for every token in training and validation modes; for the last token only in generation mode.
        return this.linearOutputLayer.update(normalizedOutput);
    }
    /**
     * Counts the trainable parameters in the whole model: the embedding, every
     * transformer block, the final layer normalization and the linear output
     * layer. Dropout has no parameters.
     *
     * The count depends on contextLength, because the position embedding has
     * one row per position. With the default Hyperparameters it only matches
     * the book's 163,009,536 when contextLength is set to GPT-2's 1024; the
     * default contextLength of 12 gives 777,216 fewer (162,232,320).
     *
     * @returns The total number of parameters in the model.
     */
    getParameterCount() {
        return (this.embedding.getParameterCount() +
            this.transformers.reduce((sum, transformer) => sum + transformer.getParameterCount(), 0) +
            this.finalLayerNormalization.getParameterCount() +
            this.linearOutputLayer.getParameterCount());
    }
}
export default GPTModel;
