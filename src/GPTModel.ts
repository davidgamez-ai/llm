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
  embedding: Embedding;

  /** The transformer blocks, applied in array order. There are numberTransformerBlocks of them. */
  transformers: Transformer[];

  /** Layer normalization applied to the output of the last transformer block, before the linear output layer. */
  finalLayerNormalization: LayerNormalization;

  /** Maps the final normalized token vectors to a score for every token in the vocabulary. */
  linearOutputLayer: LinearOutputLayer;

  /** Probability, between 0 and 1, that each embedding value is dropped (set to zero) by dropout. */
  dropoutRate: number;

  /**
   * True in training mode, when dropout is applied to the embeddings; false
   * in inference mode, when it is skipped. Read from Hyperparameters on every
   * access, so changing Hyperparameters.training after construction takes effect.
   */
  get training(): boolean {
    return this.isTraining();
  }

  /** Reads the Hyperparameters' current training flag. */
  private readonly isTraining: () => boolean;

  /** Draws from the Hyperparameters' shared random number stream, which is seeded when Hyperparameters.seed is set. */
  private readonly random: () => number;

  /**
   * Creates a GPTModel with an Embedding and a LinearOutputLayer, whose
   * matrices are built immediately, numberTransformerBlocks Transformer
   * instances and a final LayerNormalization, all built from the same
   * Hyperparameters so their dimensions match. Every component draws its
   * random values from the Hyperparameters' shared stream, so the whole
   * model is reproducible when Hyperparameters.seed is set.
   *
   * @param hyperparameters - Source of numberTransformerBlocks, dropoutRate, training and the random number stream, and passed on to Embedding, each Transformer, LayerNormalization and LinearOutputLayer. Defaults to a new Hyperparameters instance.
   */
  constructor(hyperparameters: Hyperparameters = new Hyperparameters()) {
    this.dropoutRate = hyperparameters.dropoutRate;
    this.isTraining = () => hyperparameters.training;
    this.random = () => hyperparameters.random();

    this.embedding = new Embedding(hyperparameters);
    this.embedding.build();

    this.transformers = Array.from(
      { length: hyperparameters.numberTransformerBlocks },
      () => new Transformer(hyperparameters)
    );

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
   * @returns Each row vocabularySize values wide: in training mode the logits, one row per token, in token order; in inference mode a single row of probabilities for the next token.
   */
  calculate(tokenIds: number[]): number[][] {
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
      droppedEmbeddings = embeddings.map((vector) =>
        vector.map((value) => (this.random() < this.dropoutRate ? 0 : value * keepScale))
      );

      if (DEBUG.GPT_MODEL) {
        console.log(`GPTModel| Embeddings after dropout (dropout rate ${this.dropoutRate}):`);
        if(DEBUG.VERBOSE)
          console.table(droppedEmbeddings);
      }
    }

    // Pass the embeddings through each transformer block in sequence, feeding each block's output into the next.
    let transformerOutput = droppedEmbeddings;
    for (const transformer of this.transformers) {
      transformerOutput = transformer.calculate(transformerOutput);
    }

    // Final layer normalization, applied to each token vector independently, as in GPT-2.
    const normalizedOutput = transformerOutput.map((vector) => this.finalLayerNormalization.calculate(vector));

    if (DEBUG.GPT_MODEL) {
      console.log("GPTModel| Output of final layer normalization (one row per token):");
      if(DEBUG.VERBOSE)
        console.table(normalizedOutput);
    }

    // Logits for every token in training mode; probabilities for the next token in inference mode.
    return this.linearOutputLayer.calculate(normalizedOutput);
  }

  /**
   * Counts the trainable parameters in the whole model: the embedding, every
   * transformer block, the final layer normalization and the linear output
   * layer. Dropout has no parameters.
   *
   * @returns The total number of parameters in the model.
   */
  getParameterCount(): number {
    return (
      this.embedding.getParameterCount() +
      this.transformers.reduce((sum, transformer) => sum + transformer.getParameterCount(), 0) +
      this.finalLayerNormalization.getParameterCount() +
      this.linearOutputLayer.getParameterCount()
    );
  }
}

export default GPTModel;
