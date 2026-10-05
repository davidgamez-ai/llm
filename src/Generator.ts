import BPETokenizer from "./BPETokenizer.js";
import Hyperparameters from "./Hyperparameters.js";
import GPTModel from "./GPTModel.js";
import Decoder from "./Decoder.js";

/**
 * Generates text with a GPTModel. Starting from a prompt, it repeatedly feeds
 * the text into the model, decodes the model's output into the next token's
 * text and appends it, until outputLength tokens have been generated.
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

  /** The model that produces a probability for every possible next token. */
  private readonly model: GPTModel;

  /** Converts the model's probabilities into the text of the chosen token. */
  private readonly decoder: Decoder;

  /**
   * Creates a Generator, building its own tokenizer, model and decoder.
   *
   * hyperparameters.training must be false when generate() is called;
   * otherwise the model applies dropout and the decoder throws.
   *
   * @param hyperparameters - Source of outputLength, contextLength and the settings used to build the model and decoder. Defaults to a new Hyperparameters instance.
   */
  constructor(hyperparameters: Hyperparameters = new Hyperparameters()) {
    this.hyperparameters = hyperparameters;
    this.tokenizer = new BPETokenizer();
    this.model = new GPTModel(hyperparameters);
    this.decoder = new Decoder(hyperparameters);
  }

  /**
   * Generates hyperparameters.outputLength tokens following the given text.
   *
   * @param text - The prompt to continue.
   * @returns The prompt followed by the generated text.
   * @throws Error if the model does not return exactly one probability distribution.
   * @throws Error if the hyperparameters are in training mode (thrown by Decoder).
   */
  generate(text: string): string {
    for (let t:number = 0; t < this.hyperparameters.outputLength; ++t) {
      /*
       * Only the last contextLength tokens are fed to the model, since the
       * position embedding has no rows beyond the context window. Once the
       * text grows longer than that, the earliest tokens are dropped.
       */
      const tokens:number[] = this.tokenizer.encode(text).slice(-this.hyperparameters.contextLength);

      const tokenProbs:number[][] =this.model.calculate(tokens);

      // In inference mode the model returns a single distribution over the next token.
      if (tokenProbs.length !== 1) {
        throw new Error(`Generator| Expected 1 probability distribution from the model; got ${tokenProbs.length}`);
      }

      text += this.decoder.greedy(tokenProbs[0]);
    }
    return text;
  }
}

export default Generator;
