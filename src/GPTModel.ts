import Hyperparameters from "./Hyperparameters.js";
import Embedding from "./Embedding.js";
import Transformer from "./Transformer.js";
import LayerNormalization from "./LayerNormalization.js";
import DEBUG from "./Debug.js";

/**
 * The GPT model. It converts a sequence of token ids into embeddings,
 * applies dropout to them (training mode only), then passes them through a
 * stack of transformer blocks in sequence, with the output of each block
 * becoming the input to the next. Finally, layer normalization is applied
 * to the output of the last transformer block.
 */
class GPTModel {
  /** Converts token ids into combined token and position embeddings. */
  embedding: Embedding;

  /** The transformer blocks, applied in array order. There are numberTransformerBlocks of them. */
  transformers: Transformer[];

  /** Probability, between 0 and 1, that each embedding value is dropped (set to zero) by dropout. */
  dropoutRate: number;

  /** True in training mode, when dropout is applied to the embeddings; false in inference mode, when it is skipped. */
  training: boolean;

  /**
   * Creates a GPTModel with an Embedding, whose matrices are built
   * immediately, and numberTransformerBlocks Transformer instances, all
   * built from the same Hyperparameters so their dimensions match.
   *
   * @param hyperparameters - Source of numberTransformerBlocks, dropoutRate and training, and passed on to Embedding and each Transformer. Defaults to a new Hyperparameters instance.
   * @param seed - Optional seed passed to Embedding.build so the embedding matrices are reproducible. When omitted, they are random on every run.
   */
  constructor(hyperparameters: Hyperparameters = new Hyperparameters(), seed?: number) {
    this.dropoutRate = hyperparameters.dropoutRate;
    this.training = hyperparameters.training;

    this.embedding = new Embedding(hyperparameters);
    this.embedding.build(seed);

    this.transformers = Array.from(
      { length: hyperparameters.numberTransformerBlocks },
      () => new Transformer(hyperparameters)
    );
  }

  /**
   * Runs tokenized text through the model: embedding, dropout on the
   * embeddings (training mode only), each transformer block in turn, then
   * layer normalization of the final block's output.
   *
   * @param tokenIds - Token ids of the tokenized input text, in sequence order.
   * @returns The layer normalized output of the final transformer block: one row per token, in token order, each row embeddingSize values wide.
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
        vector.map((value) => (Math.random() < this.dropoutRate ? 0 : value * keepScale))
      );

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
    const output = transformerOutput.map((vector) => LayerNormalization.calculate(vector));

    if (DEBUG.GPT_MODEL) {
      console.log("GPTModel| Output of final layer normalization (one row per token):");
      console.table(output);
    }

    return output;
  }
}

export default GPTModel;
