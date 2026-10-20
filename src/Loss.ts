/**
 * The cross entropy loss used to train the model, and its gradient: the
 * errors at the logits that start backpropagation.
 *
 * The model outputs logits (unnormalized scores), not probabilities. Softmax
 * turns a row of logits into probabilities, and the loss is the negative log
 * of the probability given to the token that actually comes next. Softmax
 * and the loss are kept together here because, combined, their gradient is
 * very simple (see crossEntropyErrors).
 */

/**
 * Converts a row of values (e.g. logits) into a probability distribution:
 * every value positive and the row summing to 1.
 *
 *     softmax(x)[i] = exp(x[i]) / sum over j of exp(x[j])
 *
 * The row's maximum is subtracted before exponentiating to avoid overflow;
 * this does not change the result because softmax is unaffected by adding a
 * constant to every input. The maximum is found with reduce rather than
 * Math.max(...values), because spreading a vocabulary-sized row as arguments
 * can exceed the engine's argument limit. A value of -Infinity (e.g. a
 * causally masked attention score) gets a probability of exactly 0.
 *
 * @param values - The values to convert. At least one must be finite.
 * @returns One probability per value, in the same order.
 */
export function softmax(values: number[]): number[] {
  const max: number = values.reduce((largest: number, value: number) => Math.max(largest, value), -Infinity);
  const exponentials: number[] = values.map((value: number) => Math.exp(value - max));
  const total: number = exponentials.reduce((sum: number, value: number) => sum + value, 0);
  return exponentials.map((value: number) => value / total);
}

/**
 * Checks that a target token id has a logit.
 *
 * @param logits - The row of logits.
 * @param target - The target token id.
 * @throws RangeError if target is not a whole number from 0 to logits.length - 1.
 */
function checkTarget(logits: number[], target: number): void {
  if (!(Number.isInteger(target) && target >= 0 && target < logits.length)) {
    throw new RangeError(`Loss| Target ${target} has no logit; expected a whole number from 0 to ${logits.length - 1}`);
  }
}

/**
 * The cross entropy loss of one prediction: the negative log of the
 * probability that softmax gives to the target token. It is 0 when the model
 * is certain of the right token, and grows without limit as the probability
 * of the right token falls towards 0. An untrained model that gives every
 * token the same probability has a loss of ln(vocabularySize).
 *
 * The log probability is calculated directly with a log-softmax, rather
 * than as log(softmax(logits)[target]), so that a tiny probability does not
 * round to 0 and give a loss of Infinity:
 *
 *     log probability = logit[target] - max - log(sum over j of exp(logit[j] - max))
 *
 * @param logits - The model's logits for one position: one per vocabulary token.
 * @param target - The id of the token that actually comes next.
 * @returns The loss: -log(probability of target).
 * @throws RangeError if target is not a valid index into logits.
 */
export function crossEntropyLoss(logits: number[], target: number): number {
  checkTarget(logits, target);

  const maxLogit: number = logits.reduce((max: number, value: number) => Math.max(max, value), -Infinity);
  const sumExp: number = logits.reduce((sum: number, value: number) => sum + Math.exp(value - maxLogit), 0);
  const logProbability: number = logits[target] - maxLogit - Math.log(sumExp);
  return -logProbability;
}

/**
 * The errors at the logits: the gradient of the cross entropy loss with
 * respect to each logit, multiplied by scale. This is where
 * backpropagation starts.
 *
 *     error[i] = scale * (softmax(logits)[i] - oneHot(target)[i])
 *
 * Why so simple? Write p = softmax(logits) and L = -log p[target]. Then
 *
 *     dL/dlogit[i] = -(1 / p[target]) * dp[target]/dlogit[i]
 *
 * and the derivative of softmax is
 *
 *     dp[target]/dlogit[i] = p[target] * (1 - p[i])   if i is the target
 *                          = -p[target] * p[i]        otherwise
 *
 * so the p[target] terms cancel, leaving p[i] - 1 for the target and p[i]
 * for every other token: the probability minus the one-hot target. This is
 * the same formula as the output layer errors in a classifier (dnn's
 * OutputLayer.calculateErrors). Every logit gets an error: the target's is
 * negative (raising it lowers the loss) and every other is positive
 * (lowering them lowers the loss). The errors always sum to 0.
 *
 * @param logits - The model's logits for one position: one per vocabulary token.
 * @param target - The id of the token that actually comes next.
 * @param scale - Multiplies every error. The trainer passes 1 / contextLength, because the loss of a window is the mean over its positions. Defaults to 1.
 * @returns One error per logit, in the same order.
 * @throws RangeError if target is not a valid index into logits.
 */
export function crossEntropyErrors(logits: number[], target: number, scale: number = 1): number[] {
  checkTarget(logits, target);

  const probabilities: number[] = softmax(logits);
  return probabilities.map((probability: number, i: number) => scale * (probability - (i === target ? 1 : 0)));
}
