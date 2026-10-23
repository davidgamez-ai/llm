import Hyperparameters from "./Hyperparameters.js";
import { softmax } from "./Loss.js";

/**
 * Chooses the next token from the model's output, an array of scores (logits
 * or probabilities) with one entry per token in the vocabulary, and returns
 * the chosen token's id. Each method implements a different strategy for choosing the token.
 * Converting ids back into text is left to the caller (e.g. Generator), so
 * that the generated ids can be decoded together once generation is finished.
 */
class Decoder {
  /**
   * The Hyperparameters this decoder was built with. A reference is kept,
   * rather than copies of its values, so that switching to generation mode
   * after construction is seen by the decoder.
   */
  private readonly hyperparameters: Hyperparameters;

  /** Draws from the Hyperparameters' shared random number stream, used to break ties and to sample tokens. */
  private readonly random: () => number;

  /**
   * Creates a Decoder. Random choices use the Hyperparameters' shared
   * random number stream, so they are reproducible when
   * Hyperparameters.seed is set and differ on every run when it is not.
   *
   * @param hyperparameters - Source of the mode, vocabularySize, samplingMethod, temperature, topK, topP and the random number stream. Defaults to a new Hyperparameters instance.
   */
  constructor(hyperparameters: Hyperparameters = new Hyperparameters()) {
    this.hyperparameters = hyperparameters;
    this.random = () => hyperparameters.random();
  }

  /**
   * Greedy decoding: picks the token with the highest probability and
   * returns its id. If several tokens share the highest probability, one
   * of them is chosen at random. Logits can be passed instead of
   * probabilities, since the highest logit is also the highest probability.
   *
   * @param probabilities - One probability per token in the vocabulary, indexed by token id. Must be vocabularySize long.
   * @returns The id of the chosen token.
   * @throws Error if the hyperparameters are not in generation mode, since text is only generated in that mode.
   * @throws RangeError if probabilities is not vocabularySize long.
   */
  greedy(probabilities: number[]): number {
    this.validate(probabilities);

    /*
     * Single pass that tracks the highest probability seen so far and every
     * token id that has it. A new maximum resets the list; an equal value
     * is added to it, so ties are collected without a second scan.
     */
    let highestProbability = -Infinity;
    let bestTokenIds: number[] = [];
    for (let tokenId = 0; tokenId < probabilities.length; tokenId++) {
      const probability = probabilities[tokenId];
      if (probability > highestProbability) {
        highestProbability = probability;
        bestTokenIds = [tokenId];
      } else if (probability === highestProbability) {
        bestTokenIds.push(tokenId);
      }
    }

    // bestTokenIds is empty only if every value is NaN, which fails every comparison.
    if (bestTokenIds.length === 0) {
      throw new RangeError("Decoder| probabilities contains no valid values");
    }

    return bestTokenIds[Math.floor(this.random() * bestTokenIds.length)];
  }

  /**
   * Chooses the next token with the method set by
   * Hyperparameters.samplingMethod: greedy(), topK() or topP(). The setting
   * is read on every call, so changing it takes effect immediately.
   *
   * @param logits - One score per token in the vocabulary, indexed by token id, as returned by the model. Must be vocabularySize long.
   * @returns The id of the chosen token.
   * @throws Error if the hyperparameters are not in generation mode.
   * @throws RangeError if logits is not vocabularySize long.
   */
  decode(logits: number[]): number {
    switch (this.hyperparameters.samplingMethod) {
      case Hyperparameters.TOP_K_SAMPLING:
        return this.topK(logits);
      case Hyperparameters.TOP_P_SAMPLING:
        return this.topP(logits);
      default:
        // Hyperparameters only accepts the three methods, so this is GREEDY_SAMPLING.
        return this.greedy(logits);
    }
  }

  /**
   * Top-k sampling: keeps only the Hyperparameters.topK tokens with the
   * highest logits and picks one of them at random, weighted by their
   * probabilities after temperature is applied (see applyTemperature). This
   * stops the many unlikely tokens, which together can have a sizeable share
   * of the probability, from ever being chosen. Tokens that tie with the k-th
   * highest logit are kept as well, as in the book. A temperature of 0 always
   * picks the most likely token, so greedy decoding is used instead.
   *
   * @param logits - One score per token in the vocabulary, indexed by token id, as returned by the model. Must be vocabularySize long.
   * @returns The id of the chosen token.
   * @throws Error if the hyperparameters are not in generation mode.
   * @throws RangeError if logits is not vocabularySize long.
   */
  topK(logits: number[]): number {
    this.validate(logits);
    if (this.hyperparameters.temperature === 0) {
      return this.greedy(logits);
    }

    // The k-th highest logit is the lowest one kept; topK may exceed the vocabulary, in which case every token is kept.
    const k: number = Math.min(this.hyperparameters.topK, logits.length);
    const threshold: number = [...logits].sort((a, b) => b - a)[k - 1];

    // A logit of -Infinity becomes a probability of exactly 0 in the softmax, so the other tokens can never be chosen.
    const keptLogits: number[] = logits.map((logit) => (logit >= threshold ? logit : -Infinity));

    return this.drawToken(this.applyTemperature(keptLogits));
  }

  /**
   * Top-p (nucleus) sampling: applies the temperature (see
   * applyTemperature), then keeps the smallest set of most likely tokens
   * whose probabilities add up to at least Hyperparameters.topP, and picks
   * one of them at random, weighted by their probabilities. Unlike top-k, the
   * number of tokens kept adapts to the model's confidence: few when one
   * continuation is clearly best, many when lots are plausible. A
   * temperature of 0 always picks the most likely token, so greedy decoding
   * is used instead.
   *
   * @param logits - One score per token in the vocabulary, indexed by token id, as returned by the model. Must be vocabularySize long.
   * @returns The id of the chosen token.
   * @throws Error if the hyperparameters are not in generation mode.
   * @throws RangeError if logits is not vocabularySize long.
   */
  topP(logits: number[]): number {
    this.validate(logits);
    if (this.hyperparameters.temperature === 0) {
      return this.greedy(logits);
    }

    const probabilities: number[] = this.applyTemperature(logits);

    // Token ids from most to least likely.
    const tokenIds: number[] = probabilities.map((_, tokenId) => tokenId);
    tokenIds.sort((a, b) => probabilities[b] - probabilities[a]);

    /*
     * Add tokens, most likely first, until together they reach topP. Every
     * other token keeps a weight of 0. The kept weights add up to topP or a
     * little more rather than 1; drawToken scales by their total.
     */
    const keptProbabilities: number[] = new Array(probabilities.length).fill(0);
    let cumulativeProbability = 0;
    for (const tokenId of tokenIds) {
      keptProbabilities[tokenId] = probabilities[tokenId];
      cumulativeProbability += probabilities[tokenId];
      if (cumulativeProbability >= this.hyperparameters.topP) {
        break;
      }
    }

    return this.drawToken(keptProbabilities);
  }

  /**
   * Divides the logits by Hyperparameters.temperature and converts them into
   * probabilities with a softmax. A temperature below 1 spreads the logits
   * further apart, so the softmax gives even more of the probability to the
   * most likely tokens and the text becomes more predictable; at 1 the
   * model's own probabilities are used unchanged. The callers handle a
   * temperature of 0, which would divide by zero.
   *
   * @param logits - The logits to convert.
   * @returns One probability per logit, adding up to 1.
   */
  private applyTemperature(logits: number[]): number[] {
    const temperature: number = this.hyperparameters.temperature;
    return softmax(logits.map((logit) => logit / temperature));
  }

  /**
   * Picks a token at random, with a chance proportional to its weight. The
   * weights are laid end to end, a random point is chosen along their total
   * length, and the token whose stretch contains it is picked. The weights
   * need not add up to 1, so the probabilities kept by top-p can be passed
   * without rescaling them.
   *
   * @param weights - One weight (0 or more) per token, indexed by token id.
   * @returns The id of the chosen token.
   * @throws RangeError if no weight is greater than 0.
   */
  private drawToken(weights: number[]): number {
    const total: number = weights.reduce((sum, weight) => sum + (weight > 0 ? weight : 0), 0);
    const target: number = this.random() * total;

    let cumulativeWeight = 0;
    let lastPossibleTokenId = -1;
    for (let tokenId = 0; tokenId < weights.length; tokenId++) {
      if (weights[tokenId] > 0) {
        cumulativeWeight += weights[tokenId];
        lastPossibleTokenId = tokenId;
        if (target < cumulativeWeight) {
          return tokenId;
        }
      }
    }

    // Rounding can leave the running total just under the target; the last possible token then takes the remainder.
    if (lastPossibleTokenId === -1) {
      throw new RangeError("Decoder| logits contains no valid values");
    }
    return lastPossibleTokenId;
  }

  /**
   * Checks that decoding is allowed and that the probabilities cover the
   * whole vocabulary.
   *
   * @param probabilities - The probabilities to check.
   * @throws Error if the hyperparameters are not in generation mode.
   * @throws RangeError if probabilities is not vocabularySize long.
   */
  private validate(probabilities: number[]): void {
    if (this.hyperparameters.mode !== Hyperparameters.GENERATION_MODE) {
      throw new Error("Decoder| Mode must be GENERATION_MODE to decode tokens");
    }
    if (probabilities.length !== this.hyperparameters.vocabularySize) {
      throw new RangeError(
        `Decoder| Expected ${this.hyperparameters.vocabularySize} probabilities; got ${probabilities.length}`
      );
    }
  }
}

export default Decoder;
