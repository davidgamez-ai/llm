import Hyperparameters from "./Hyperparameters.js";
import MultiHeadAttention from "./MultiHeadAttention.js";
import FeedForward from "./FeedForward.js";
import LayerNormalization from "./LayerNormalization.js";
import DEBUG from "./Debug.js";
/**
 * A single GPT transformer block. It combines multi-head attention, which
 * lets each token gather information from the other tokens in the
 * sequence, with a feed forward network, which transforms each token
 * independently. Layer normalization is applied to the input of each of
 * these two stages, and dropout to the output of each, as in the book's
 * TransformerBlock (drop_shortcut):
 *
 * 1. Layer normalization of each token embedding.
 * 2. Multi-head attention over the normalized embeddings.
 * 3. Dropout on the attention output (training mode only).
 * 4. A shortcut connection that adds the input embeddings to the output of step 3.
 * 5. Layer normalization of each resulting vector.
 * 6. The feed forward network, applied to each normalized vector.
 * 7. Dropout on the feed forward output (training mode only).
 * 8. A shortcut connection that adds the result of step 4 to the output of step 7.
 *
 * This dropout on the attention output is separate from the dropout that
 * GPTAttention applies to its attention weights: that one acts on how much
 * each token attends to the others, inside each head, whereas this one acts
 * on the attention stage's final output, just before it is added to the
 * shortcut.
 */
class Transformer {
    /** The multi-head attention stage of the block. */
    multiHeadAttention;
    /** The feed forward network stage of the block. */
    feedForward;
    /** Layer normalization applied to the block's input, before multi-head attention. */
    attentionLayerNormalization;
    /** Layer normalization applied to the attention stage's output, before the feed forward network. */
    feedForwardLayerNormalization;
    /** Probability, between 0 and 1, that each value of the attention output and the feed forward output is dropped (set to zero) by dropout. */
    dropoutRate;
    /**
     * The current mode: Hyperparameters.TRAINING_MODE, VALIDATION_MODE or
     * GENERATION_MODE. Dropout is only applied to the attention output and the
     * feed forward output in TRAINING_MODE. Read from Hyperparameters on every access, so changing
     * Hyperparameters.mode after construction takes effect.
     */
    get mode() {
        return this.getMode();
    }
    /** Reads the Hyperparameters' current mode. */
    getMode;
    /** Draws from the Hyperparameters' shared random number stream, which is seeded when Hyperparameters.seed is set. */
    random;
    /**
     * Creates a Transformer with its own MultiHeadAttention, FeedForward and
     * two LayerNormalization instances, all built from the same
     * Hyperparameters so their dimensions match, and copies dropoutRate
     * from them. Each layer normalization has its own trainable
     * scale and shift, so the two are separate instances.
     *
     * @param hyperparameters - Source of dropoutRate, mode and the random number stream, and passed on to MultiHeadAttention, FeedForward and LayerNormalization. Defaults to a new Hyperparameters instance.
     */
    constructor(hyperparameters = new Hyperparameters()) {
        this.dropoutRate = hyperparameters.dropoutRate;
        this.getMode = () => hyperparameters.mode;
        this.random = () => hyperparameters.random();
        this.multiHeadAttention = new MultiHeadAttention(hyperparameters);
        this.feedForward = new FeedForward(hyperparameters);
        this.attentionLayerNormalization = new LayerNormalization(hyperparameters);
        this.feedForwardLayerNormalization = new LayerNormalization(hyperparameters);
    }
    /**
     * Runs the token embeddings through the transformer block: layer
     * normalization, multi-head attention, dropout (training mode only), a
     * shortcut connection adding the input embeddings to the attention
     * output, layer normalization, the feed forward network, dropout
     * (training mode only), then a second shortcut connection adding the
     * attention stage's output to the feed forward output. Layer
     * normalization and the feed forward network are applied to each token
     * vector separately.
     *
     * @param embeddings - Token embeddings: one row per token, each row embeddingSize values wide.
     * @returns The output of the block: one row per token, in token order, each row embeddingSize values wide.
     */
    calculate(embeddings) {
        // Normalize each token embedding independently before attention.
        const normalizedEmbeddings = embeddings.map((embedding) => this.attentionLayerNormalization.calculate(embedding));
        const attentionOutput = this.multiHeadAttention.calculate(normalizedEmbeddings);
        // Dropout on the attention output before it is added to the shortcut (training mode only).
        const droppedAttentionOutput = this.dropout(attentionOutput, "Attention output");
        /*
         * Shortcut (residual) connection: add the original, unnormalized input
         * embeddings to the attention output, element by element. This gives
         * gradients a direct path back through the block during training, and
         * means attention only has to learn a change to each embedding rather
         * than a complete replacement for it.
         */
        const shortcutOutput = this.add(droppedAttentionOutput, embeddings);
        // Normalize each shortcut output vector independently before the feed forward network.
        const normalizedShortcutOutput = shortcutOutput.map((vector) => this.feedForwardLayerNormalization.calculate(vector));
        // The feed forward network processes one token vector at a time.
        const feedForwardOutput = normalizedShortcutOutput.map((vector) => this.feedForward.calculate(vector));
        // Dropout on the feed forward output before it is added to the shortcut (training mode only).
        const droppedFeedForwardOutput = this.dropout(feedForwardOutput, "Feed forward output");
        /*
         * Second shortcut connection: add the output of the attention stage
         * (including the first shortcut) to the feed forward output, element by
         * element. As in GPT-2, this keeps an unbroken path from the block's
         * input to its output, so the feed forward network only has to learn a
         * change to each token vector.
         */
        const output = this.add(droppedFeedForwardOutput, shortcutOutput);
        if (DEBUG.TRANSFORMER) {
            console.log("Transformer| Output (one row per token):");
            console.table(output);
        }
        return output;
    }
    /**
     * Adds two matrices of token vectors element by element, as a shortcut
     * connection does: output[token][index] = first[token][index] + second[token][index].
     *
     * @param first - One row per token, each row embeddingSize values wide.
     * @param second - The same shape as first.
     * @returns A new matrix of the same shape holding the sums.
     */
    add(first, second) {
        const sum = [];
        for (let token = 0; token < first.length; token++) {
            sum.push([]);
            for (let index = 0; index < first[token].length; index++) {
                sum[token].push(first[token][index] + second[token][index]);
            }
        }
        return sum;
    }
    /**
     * Applies dropout to a stage's output, only in training mode. Each value
     * is set to 0 with probability dropoutRate; the values that are kept are
     * scaled by 1 / (1 - dropoutRate) so the expected size of each value is
     * unchanged, as in GPTAttention. In validation and generation modes the
     * values are returned unchanged.
     *
     * @param values - The stage's output: one row per token, each row embeddingSize values wide.
     * @param label - Name of the stage, used in debug output.
     * @returns A new matrix with dropout applied in training mode; otherwise values itself.
     */
    dropout(values, label) {
        if (this.mode !== Hyperparameters.TRAINING_MODE) {
            return values;
        }
        const keepScale = 1 / (1 - this.dropoutRate);
        const dropped = values.map((vector) => vector.map((value) => (this.random() < this.dropoutRate ? 0 : value * keepScale)));
        if (DEBUG.TRANSFORMER) {
            console.log(`Transformer| ${label} after dropout (dropout rate ${this.dropoutRate}):`);
            console.table(dropped);
        }
        return dropped;
    }
    /**
     * Counts the trainable parameters in the block: those of the multi-head
     * attention stage, the feed forward network and both layer
     * normalizations. The dropouts and the shortcut connections have no parameters.
     *
     * @returns The total number of parameters in the block.
     */
    getParameterCount() {
        return (this.multiHeadAttention.getParameterCount() +
            this.feedForward.getParameterCount() +
            this.attentionLayerNormalization.getParameterCount() +
            this.feedForwardLayerNormalization.getParameterCount());
    }
}
export default Transformer;
