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
     * Backpropagation through the whole model: passes the errors at the
     * logits back through every component, in the reverse order of update():
     *
     *     linear output layer → final layer normalization → transformer blocks
     *     (last to first) → embedding dropout → embedding
     *
     * Each component turns the errors at its output into the errors at its
     * input, and stores what it needs to calculate its gradients. The model
     * takes the errors rather than the targets, so it contains no loss
     * calculation: the trainer works out the errors from the loss (with
     * crossEntropyErrors) and hands them in, as dnn's trainer does.
     *
     * Must follow an update() in training or validation mode, which return
     * logits for every token; in generation mode only the last token's logits
     * are calculated, so there is nothing to pass errors back through for the
     * other tokens.
     *
     * @param outputErrors - The errors at the logits: one row per token, each vocabularySize values wide.
     * @throws RangeError if outputErrors does not have one row per token given to the most recent update().
     */
    backpropagate(outputErrors) {
        let errors = this.linearOutputLayer.backpropagate(outputErrors);
        errors = this.finalLayerNormalization.backpropagate(errors);
        // The blocks were applied first to last, so the errors pass back through them last to first.
        for (let block = this.transformers.length - 1; block >= 0; block--) {
            errors = this.transformers[block].backpropagate(errors);
        }
        errors = this.embeddingDropout.backpropagate(errors);
        this.embedding.backpropagate(errors);
        if (DEBUG.GPT_MODEL) {
            console.log("GPTModel| Backpropagation complete.");
        }
    }
    /**
     * Calculates the gradients of every trainable parameter in the model, by
     * asking each component to calculate its own from the values it stored
     * during update() and backpropagate(). The dropouts have no parameters.
     *
     * @param batchAverage - False for the first window of a batch, which replaces the stored gradients; true for later windows, which are averaged into them.
     */
    calculateGradients(batchAverage) {
        this.embedding.calculateGradients(batchAverage);
        for (const transformer of this.transformers) {
            transformer.calculateGradients(batchAverage);
        }
        this.finalLayerNormalization.calculateGradients(batchAverage);
        this.linearOutputLayer.calculateGradients(batchAverage);
    }
    /**
     * Updates every trainable parameter in the model from the gradients
     * calculated by calculateGradients(): one optimization step. Called once
     * per batch, after the gradients of every window in the batch have been
     * calculated and averaged. As in dnn, the optimizer's step() is called
     * first, then each component updates its own parameters.
     *
     * @param optimizer - The AdamW optimizer, shared by every component and kept for the whole of training, so its running averages carry over from batch to batch.
     * @throws RangeError if calculateGradients() has not been called, since there are then no gradients to update from.
     */
    updateWeights(optimizer) {
        optimizer.step();
        this.embedding.updateWeights(optimizer);
        for (const transformer of this.transformers) {
            transformer.updateWeights(optimizer);
        }
        this.finalLayerNormalization.updateWeights(optimizer);
        this.linearOutputLayer.updateWeights(optimizer);
    }
    /**
     * Counts the trainable parameters in the whole model: the embedding, every
     * transformer block, the final layer normalization and the linear output
     * layer. Dropout has no parameters.
     *
     * With the default Hyperparameters (GPT-2 small) the count is 163,037,184.
     * Subtracting the linear output layer's 38,597,376 gives GPT-2's
     * 124,439,808: GPT-2 ties the output layer to the token embedding, so it
     * does not count those weights twice. The book's 163,009,536 is 27,648
     * fewer, because it turns off the query, key and value biases
     * (qkvBias false: 3 x 768 biases in each of the 12 blocks). The count also
     * depends on contextLength, because the position embedding has one row
     * per position.
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
