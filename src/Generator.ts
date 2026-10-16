import BPETokenizer from "./BPETokenizer.js";
import Hyperparameters from "./Hyperparameters.js";
import GPTModel from "./GPTModel.js";
import Decoder from "./Decoder.js";

/**
 * Generates text with a GPTModel. Starting from a prompt, it repeatedly feeds
 * the text into the model, converts the logits the model returns into
 * probabilities, decodes those into the next token's text and appends it,
 * until outputLength tokens have been generated.
 *
 * Tokens are currently chosen with greedy decoding.
 */
class Generator {
  /**
   * The Hyperparameters this generator was built with. A reference is kept,
   * rather than copies of its values, so that changes to outputLength after
   * construction are seen by generate().
   */
  private readonly hyperparameters: Hyperparameters;

  /** Converts text into the token ids fed to the model. */
  private readonly tokenizer: BPETokenizer;

  /** The model that produces a score (logit) for every possible next token. */
  private readonly model: GPTModel;

  /** Converts the probabilities into the text of the chosen token. */
  private readonly decoder: Decoder;

  /**
   * Creates a Generator for the given model, building its own tokenizer and
   * decoder. The model is passed in rather than built here so that the same
   * instance can be shared with a Trainer, and text is generated with the
   * trained weights.
   *
   * hyperparameters.mode must be Hyperparameters.GENERATION_MODE when
   * generate() is called; otherwise the decoder throws.
   *
   * @param hyperparameters - The settings the model was built with; also the source of outputLength, contextLength and the settings used to build the decoder.
   * @param model - The model used to generate text. Must have been built from the same hyperparameters.
   */
  constructor(hyperparameters: Hyperparameters, model: GPTModel) {
    this.hyperparameters = hyperparameters;
    this.tokenizer = new BPETokenizer();
    this.model = model;
    this.decoder = new Decoder(hyperparameters);
  }

  /**
   * Generates hyperparameters.outputLength tokens following the given text.
   *
   * @param text - The prompt to continue.
   * @returns The prompt followed by the generated text.
   * @throws Error if the hyperparameters are not in generation mode (thrown by Decoder).
   */
  generate(text: string): string {
    for (let t:number = 0; t < this.hyperparameters.outputLength; ++t) {
      /*
       * Only the last contextLength tokens are fed to the model, since the
       * position embedding has no rows beyond the context window. Once the
       * text grows longer than that, the earliest tokens are dropped.
       */
      const tokens:number[] = this.tokenizer.encode(text).slice(-this.hyperparameters.contextLength);

      const logits:number[][] = this.model.calculate(tokens);

      /*
       * Only the last row of logits is needed: it scores every candidate for
       * the token that follows the whole input. In generation mode the model
       * only returns that row, but taking the last row would also work if it
       * returned logits for every token.
       */
      const probabilities:number[] = Generator.softmax(logits[logits.length - 1]);

      text += this.decoder.greedy(probabilities);
    }
    return text;
  }

  /**
   * Converts a row of logits into a probability distribution over the
   * vocabulary: every value positive and the row summing to 1.
   *
   * The row's maximum is subtracted before exponentiating to avoid
   * overflow; this does not change the result because softmax is unaffected
   * by adding a constant to every input. The maximum is found with reduce
   * rather than Math.max(...logits), because spreading a vocabulary-sized row
   * as arguments can exceed the engine's argument limit.
   *
   * @param logits - One logit per token in the vocabulary, indexed by token id.
   * @returns One probability per token in the vocabulary, indexed by token id.
   */
  private static softmax(logits: number[]): number[] {
    const maxLogit:number = logits.reduce((max:number, value:number) => Math.max(max, value), -Infinity);
    const exponentiated:number[] = logits.map((value:number) => Math.exp(value - maxLogit));
    const total:number = exponentiated.reduce((sum:number, value:number) => sum + value, 0);
    return exponentiated.map((value:number) => value / total);
  }
}

export default Generator;
