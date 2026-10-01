import Hyperparameters from "./Hyperparameters.js";
import Embedding from "./Embedding.js";
import Transformer from "./Transformer.js";
import LayerNormalization from "./LayerNormalization.js";
import LinearOutputLayer from "./LinearOutputLayer.js";
import DEBUG from "./Debug.js";
/**
 * The GPT model. It converts a sequence of token ids into embeddings,
 * applies dropout to them (training mode only), then passes them through a
 * stack of transformer blocks in sequence, with the output of each block
 * becoming the input to the next. Layer normalization is then applied to the
 * output of the last transformer block, and the linear output layer turns
 * the result into logits (training mode) or a probability distribution over
 * the next token (inference mode).
 */
class GPTModel {
    /** Converts token ids into combined token and position embeddings. */
    embedding;
    /** The transformer blocks, applied in array order. There are numberTransformerBlocks of them. */
    transformers;
    /** Maps the final normalized token vectors to a score for every token in the vocabulary. */
    linearOutputLayer;
    /** Probability, between 0 and 1, that each embedding value is dropped (set to zero) by dropout. */
    dropoutRate;
    /** True in training mode, when dropout is applied to the embeddings; false in inference mode, when it is skipped. */
    training;
    /**
     * Creates a GPTModel with an Embedding and a LinearOutputLayer, whose
     * matrices are built immediately, and numberTransformerBlocks Transformer
     * instances, all built from the same Hyperparameters so their dimensions
     * match.
     *
     * @param hyperparameters - Source of numberTransformerBlocks, dropoutRate and training, and passed on to Embedding, each Transformer and LinearOutputLayer. Defaults to a new Hyperparameters instance.
     * @param seed - Optional seed passed to Embedding.build and LinearOutputLayer.build so their matrices are reproducible. When omitted, they are random on every run.
     */
    constructor(hyperparameters = new Hyperparameters(), seed) {
        this.dropoutRate = hyperparameters.dropoutRate;
        this.training = hyperparameters.training;
        this.embedding = new Embedding(hyperparameters);
        this.embedding.build(seed);
        this.transformers = Array.from({ length: hyperparameters.numberTransformerBlocks }, () => new Transformer(hyperparameters));
        /*
         * The output layer's weight matrix has the same dimensions as the
         * embedding's token matrix and is built the same way, so the same seed
         * would make the two matrices identical. Offsetting the seed keeps them
         * reproducible but independent.
         */
        this.linearOutputLayer = new LinearOutputLayer(hyperparameters);
        this.linearOutputLayer.build(seed === undefined ? undefined : seed + 1);
    }
    /**
     * Runs tokenized text through the model: embedding, dropout on the
     * embeddings (training mode only), each transformer block in turn, layer
     * normalization of the final block's output, then the linear output layer.
     *
     * @param tokenIds - Token ids of the tokenized input text, in sequence order.
     * @returns Each row vocabularySize values wide: in training mode the logits, one row per token, in token order; in inference mode a single row of probabilities for the next token.
     */
    calculate(tokenIds) {
        const embeddings = this.embedding.getEmbeddingFromTokens(tokenIds);
        /*
         * Dropout on the embeddings, only in training mode. Each value is set to
         * 0 with probability dropoutRate; the values that are kept are scaled by
         * 1 / (1 - dropoutRate) so the expected size of each value is unchanged,
         * as in Transformer. In inference mode the embeddings are used unchanged.
         */
        let droppedEmbeddings = embeddings;
        if (this.training) {
            const keepScale = 1 / (1 - this.dropoutRate);
            droppedEmbeddings = embeddings.map((vector) => vector.map((value) => (Math.random() < this.dropoutRate ? 0 : value * keepScale)));
            if (DEBUG.GPT_MODEL) {
                console.log(`GPTModel| Embeddings after dropout (dropout rate ${this.dropoutRate}):`);
                console.table(droppedEmbeddings);
            }
        }
        // Pass the embeddings through each transformer block in sequence, feeding each block's output into the next.
        let transformerOutput = droppedEmbeddings;
        for (const transformer of this.transformers) {
            transformerOutput = transformer.calculate(transformerOutput);
        }
        // Final layer normalization, applied to each token vector independently, as in GPT-2.
        const normalizedOutput = transformerOutput.map((vector) => LayerNormalization.calculate(vector));
        if (DEBUG.GPT_MODEL) {
            console.log("GPTModel| Output of final layer normalization (one row per token):");
            console.table(normalizedOutput);
        }
        // Logits for every token in training mode; probabilities for the next token in inference mode.
        return this.linearOutputLayer.calculate(normalizedOutput);
    }
    /**
     * Counts the trainable parameters in the whole model: the embedding, every
     * transformer block and the linear output layer. Dropout and the final
     * layer normalization have no parameters.
     *
     * @returns The total number of parameters in the model.
     */
    getParameterCount() {
        return (this.embedding.getParameterCount() +
            this.transformers.reduce((sum, transformer) => sum + transformer.getParameterCount(), 0) +
            this.linearOutputLayer.getParameterCount());
    }
}
export default GPTModel;
