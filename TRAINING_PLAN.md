# Plan: adding training (backpropagation) to the GPT model

This plan adds backpropagation of errors, gradient calculation and AdamW
weight updates to the GPT model, so that `Trainer.train()` changes the
weights and the loss falls.

It is written as a sequence of small steps. Each step:

- changes only a few files;
- leaves the code compiling and **all tests passing** (`npm test`);
- ends with a **check**: a test or run whose expected result is stated, so
  you know the step is correct before moving on.

Part A sets out the conventions every step follows. Part B is the steps.
Part C is reference material (dnn mapping, configuration, optional extras).

---

# Part A: Conventions

## A1. Design principles (following dnn)

The design follows [davidgamez-ai/dnn](https://github.com/davidgamez-ai/dnn):

- **Each component stores what it needs for backpropagation.** During
  `update()` (the forward pass) it keeps its inputs and intermediate values.
  During backpropagation it stores the errors it receives.
- **Errors flow backwards** through the components in the reverse order of
  the forward pass. Each component turns the errors at its output into the
  errors at its input.
- **Gradients are calculated locally** in each component, from its own stored
  inputs and errors, in a separate `calculateGradients(batchAverage)` step.
- **One shared `AdamW` optimizer** updates every component once per batch
  through `updateWeights(optimizer)`.
- **No automatic differentiation.** Every derivative is written by hand in the
  component it belongs to, with a comment giving the formula.

Clarity comes before performance throughout.

## A2. Method interface for every component

```ts
update(inputs: number[][]): number[][]               // forward pass; stores inputs and intermediates
backpropagate(outputErrors: number[][]): number[][]  // stores errors; returns errors at the inputs
calculateGradients(batchAverage: boolean): void      // local gradients from stored inputs and errors
updateWeights(optimizer: AdamW): void                // AdamW step on this component's parameters
```

- `backpropagate` combines dnn's `calculateErrors` and
  `calculatePreviousLayerErrors`. LLM components have several internal error
  arrays (e.g. the errors of the queries, keys and values), so one method that
  stores them all and returns the input errors is clearer.
- Components without parameters (`Dropout`, `Transformer`, `GPTModel`) still
  implement these methods where relevant, delegating to their children.
  `Dropout` (step 2) already has `update` and `backpropagate`.
- Stored inputs, intermediates and errors are `number[][]` (one row per
  token), because every component processes a whole window of tokens.

## A3. Gradient storage and batch averaging

- Each weight `Matrix` gets a gradient `Matrix` of the same dimensions; each
  `Float32Array` of biases (or layer-norm scale/shift) gets a `Float32Array`
  gradient of the same length. They are named after the parameter, e.g.
  `queryWeightGradients`, `firstLayerBiasGradients`.
- Gradients for one window are **summed over every token position**, because
  the same weights are used at every position.
- Gradient arrays are **empty until `calculateGradients` is first called**
  (allocated lazily), so a model that is only used for generation does not
  double its memory.
- Batch averaging copies dnn exactly. Each component with parameters keeps a
  public `gradientCount`:
  - `calculateGradients(false)` (first window of a batch) replaces the stored
    gradients and sets the count to 1;
  - `calculateGradients(true)` (remaining windows) increments the count to `k`
    and folds the new gradients into a running mean:
    `mean += (new − mean) / k`.
  - Every component does this with the shared helper
    `runningMean(mean, values, count)` in `Matrix.ts` (added in step 6), so
    each `calculateGradients` ends with the same two lines:

    ```ts
    this.gradientCount = batchAverage && this.gradientCount > 0 ? this.gradientCount + 1 : 1;
    this.weightGradients = runningMean(this.weightGradients, weightGradients, this.gradientCount);
    ```
- The output errors for a window are the gradient of that window's **mean**
  loss per token: `(softmax − oneHot) / contextLength`. Every window has
  exactly `contextLength` tokens, so the mean of the window means is the mean
  over every token in the batch: the loss the trainer already reports.

## A4. Notation used in the steps

- `dX` means "the errors of X", i.e. ∂loss/∂X.
- `Σ_t` sums over the token positions in the window.
- `a ⊗ b` is the outer product: `(a ⊗ b)(row, column) = a[row] · b[column]`.
- `⊙` is element-by-element multiplication.

## A5. Fan-out rule

Where one value feeds several places (a shortcut connection; the same
embeddings feeding every attention head; one normalized vector feeding the
query, key and value projections), **the errors from each place are added
together**. Every place this happens gets a comment saying so: it is the key
idea behind backpropagation through residual networks.

---

# Part B: Steps

## Step 0: Process whole sequences in `LayerNormalization` and `FeedForward` ✅ done

`Transformer` and `GPTModel` used to call these once per token, so stored
values would only have been kept for the last token. Both now take
`update(vectors: number[][]): number[][]`, and their callers pass the whole
sequence. All 81 existing tests pass.

---

## Step 1: Record reference logits (refactor safety net) ✅ done

**Goal:** detect any accidental change to the forward pass made by later
refactoring steps.

**What was done**
- New `src/test/GPTModel.test.ts`: builds a tiny seeded model
  (`embeddingSize 8`, `2 heads`, `1 block`, `contextLength 4`, seed 42), runs
  `update()` on the token ids `[464, 3290, 318, 3290]` (a repeated token is
  included) in each of the three modes, and compares the logits with values
  recorded before any training code was added.
- Each logit row is 50,257 values wide, too many to store, so each row is
  summarized by its sum, its sum of squares and the logits at six fixed
  indices. The tolerance is `1e-12`, so only a harmless reordering of
  floating-point additions could pass.
- Training mode is included because dropout uses the random stream: if a
  later step changes the order of random draws, this test fails.
- `src/test/Trainer.test.ts` also gained a reference for the untrained
  model's validation loss (`10.81001060108122`), to check step 3.
- If the forward pass is ever changed on purpose, the reference values must
  be regenerated (the test file explains how).

**Check:** `npm test` passes (85 tests).

---

## Step 2: Extract a `Dropout` class ✅ done

**Goal:** dropout was written three times (`GPTModel`,
`Transformer.dropout`, `GPTAttention`) and its masks were thrown away. The
backward pass needs each mask, so dropout is now a small component that
keeps it.

**What was done**
- New `src/Dropout.ts`:
  - `update(values: number[][]): number[][]`: in `TRAINING_MODE` builds and
    stores a mask of `0` or `1 / (1 − dropoutRate)`, drawing one random
    number per value, row by row (the same order as before), and returns
    `values ⊙ mask`. In other modes it clears the mask and returns `values`
    unchanged.
  - `backpropagate(outputErrors: number[][]): number[][]`: returns
    `outputErrors ⊙ mask`, so dropped values get no error back. With no
    stored mask, it returns the errors unchanged. Throws `RangeError` if the
    errors are not the shape of the mask.
  - A read-only `mask` getter (`null` outside training mode), used for debug
    output and tests.
- Instances (public fields):
  - `GPTModel.embeddingDropout`;
  - `Transformer.attentionOutputDropout` and
    `Transformer.feedForwardOutputDropout`; the private `dropout` method is
    now `applyDropout(dropout, values, label)`, which also does the debug
    logging;
  - `GPTAttention.attentionDropout`.
- The `dropoutRate`, `mode`, `getMode` and `random` members that only
  existed for dropout were removed from `GPTModel` and `Transformer`, and
  `dropoutRate`, `mode` and `getMode` from `GPTAttention` (it still uses
  `random` for weight initialization).

**Check**
- New `src/test/Dropout.test.ts` (7 tests): unchanged values in validation
  and generation modes; every value is either 0 or 2 × input with a dropout
  rate of 0.5; the stored mask matches the output; `backpropagate` zeroes
  exactly the errors of dropped values and doubles the rest; errors pass
  through unchanged after a non-training `update`; wrong shapes are
  rejected; a dropout rate of 0 changes nothing.
- **Step 1's reference logits are unchanged in all three modes.**

---

## Step 3: Move the loss into `Loss.ts` and add its gradient ✅ done

**Goal:** have one place that calculates both the cross entropy loss and the
errors at the logits.

**What was done**
- New `src/Loss.ts`:
  - `softmax(values: number[]): number[]`: numerically stable, and
    `-Infinity` gives exactly 0. It replaced the private
    `Generator.softmax` and the inline softmax in `GPTAttention`, so there is
    now one implementation.
  - `crossEntropyLoss(logits: number[], target: number): number`: the
    log-softmax code moved from `Trainer.calculateTotalLoss`.
  - `crossEntropyErrors(logits: number[], target: number, scale = 1): number[]`:
    `scale · (softmax(logits) − oneHot(target))`. The comment derives why
    softmax followed by cross entropy has this simple gradient.
  - Both loss functions throw `RangeError` for a target with no logit.
- `Trainer.calculateTotalLoss` calls `crossEntropyLoss`.

**Check**
- New `src/test/Loss.test.ts` (11 tests), including a **finite-difference
  check** that `crossEntropyErrors` matches the numerical gradient of
  `crossEntropyLoss` to `1e-8`.
- The untrained validation loss and the reference logits are unchanged.

---

## Step 4: `Matrix` helpers for gradients ✅ done

**Goal:** the two operations every gradient calculation needs.

**What was done** in `src/Matrix.ts`:
- `addOuterProduct(gradients: Matrix, input, errors): void`:
  `gradients(row, column) += input[row] · errors[column]`. This is dnn's
  "weight gradient = error × input"; calling it once per token sums over
  the tokens.
- `addToVector(target: Float32Array, values): void`: `target[i] += values[i]`,
  for bias gradients.
- Both throw `RangeError` on mismatched lengths.
- A comment block, "Backpropagation through a linear layer", explains both
  helpers and includes the table showing that the two existing
  multiplication functions are each other's transpose:

| Layer | Forward | Backward (errors to inputs) |
| --- | --- | --- |
| Weights stored input × output (`GPTAttention`, `MultiHeadAttention`, `FeedForward`) | `vectorTimesMatrix(x, W)` | `matrixTimesVector(W, errors)` |
| Weights stored output × input (`LinearOutputLayer`) | `matrixTimesVector(W, x)` | `vectorTimesMatrix(errors, W)` |

**Check:** 5 new tests in `src/test/Matrix.test.ts`: hand-calculated
examples for both helpers (including accumulation across two calls),
rejection of wrong lengths, and `matrixTimesVector(W, e)` equals
`vectorTimesMatrix(e, Wᵀ)` for a hand-built transpose.

---

## Step 5: Gradient-checking test helper ✅ done

**Goal:** a reusable test that compares hand-written gradients with numerical
ones. Every later backpropagation step is checked with it.

**What was done**
- New `src/test/GradientCheck.ts` (a helper, not a test file, so the test
  runner does not run it directly) exporting
  `checkGradients(options): number`, which returns how many values it
  checked. `options`:
  - `forward(): number[][]`: runs the component's `update`. It must read the
    current contents of `inputs` and the parameters, which are changed in
    place, and must be deterministic.
  - `backward(outputErrors: number[][]): void`: typically `backpropagate`
    then `calculateGradients(false)`.
  - `parameters`: a list of `{ name, values: Float32Array, gradients: () => ArrayLike<number> }`.
  - optionally `inputs: number[][]` and `inputErrors: () => number[][]`, to
    check the errors returned by `backpropagate` too.
  - optionally `samplesPerArray` (20), `step` (1e-2), `relativeTolerance`
    (1e-2), `absoluteTolerance` (1e-5) and `seed` (1).
- How it works:
  1. Uses the made-up loss `L = Σ output ⊙ R` for a fixed random `R`, so the
     output errors are just `R`. Runs `forward()` then `backward(R)`, and
     **copies** the analytic gradients (later forward passes overwrite the
     component's stored values).
  2. For each sampled value: sets it to `v + h`, measures `L`, sets it to
     `v − h`, measures `L`, restores `v`. For parameters, the step is the
     difference of the **values read back** from the `Float32Array`,
     because 32-bit storage rounds `v ± h`.
  3. Fails with a list of up to 10 mismatches (name, index, analytic,
     numerical) if any disagree.

**Check:** `src/test/GradientCheck.test.ts` (6 tests) uses a fake linear
layer (optionally followed by tanh) with hand-written gradients built from
`addOuterProduct` and `matrixTimesVector`. The helper passes the correct
gradients, **fails** when the weight gradients are doubled, **fails** when
the input errors are 10% off, leaves the parameters and inputs unchanged,
and samples large arrays.

**State after step 5:** 114 tests, all passing.

---

## Notes that apply to steps 6–12 ✅ done

- **Pattern.** Every component now has `backpropagate(outputErrors)`, which
  stores the errors and returns the errors at its input, and
  `calculateGradients(batchAverage)`, which builds this window's gradients in
  new arrays and combines them with the stored ones through `runningMean`
  (see A3).
- **Stored state** is private. Gradients and `gradientCount` are public
  fields, so the optimizer (step 15) and the tests can read them.
- **Errors for the wrong number of tokens.** Every `backpropagate` throws a
  `RangeError` if it is not given one row of errors per token from the most
  recent `update`.
- **Forward pass unchanged.** The step 1 reference logits and the untrained
  validation loss still match after every step.
- **Test set-up.** The tests use embeddingSize 8, 4 tokens, dropout 0 unless
  testing dropout, and a shared `randomMatrix(rows, columns, seed, scale)`
  helper (added to `GradientCheck.ts`). Two choices keep the gradient checks
  meaningful:
  - The attention, multi-head attention and transformer tests set
    `linearLayerStandardDeviation = 0.5` (instead of 0.02). With GPT-2's
    small weights the attention weights are almost uniform, so the softmax
    would barely be tested.
  - Layer-norm scale and shift, and biases, are set to random values instead
    of their initial 1s and 0s, so the general case is covered.
- **Mutation check.** To confirm the gradient tests really catch mistakes,
  three bugs were injected one at a time into the compiled code, and each
  made its component's tests fail:
  - the `/ sqrt(d)` in attention's backward pass removed;
  - the variance term removed from layer normalization's backward pass;
  - the first shortcut dropped from the transformer's backward pass.

  The compiled code was then rebuilt from the unchanged source.

---

## Step 6: `LinearOutputLayer` backpropagation ✅ done

**Forward:** `z[t] = W · h[t]` (W is vocabSize × embeddingSize, no bias).

**What was done**
- New `runningMean(mean, values, count)` in `Matrix.ts`, used by every
  component for batch averaging (A3).
- `update` stores `inputs` (h).
- `backpropagate(dZ)`: stores `dZ`; returns `dH[t] = vectorTimesMatrix(dZ[t], W)`.
- `calculateGradients(batchAverage)`: `dW = Σ_t dZ[t] ⊗ h[t]`, via
  `addOuterProduct(dW, dZ[t], h[t])`. The arguments are the other way round
  from the input × output layers, because this layer's weights are stored
  output × input; a comment says so.
- Public fields: `weightGradients`, `gradientCount`.

**Check:** `LinearOutputLayer.test.ts` (3 tests): gradient check of the
weights (sampled, since there are 50,257 × 8) and the input errors; a
**batch-averaging test** showing that `calculateGradients(false)` then
`calculateGradients(true)` gives the mean of the two windows' separate
gradients, and that `false` resets the count to 1; rejection of errors for
the wrong number of tokens.

---

## Step 7: `LayerNormalization` backpropagation ✅ done

**Forward (per token):** `μ`, `σ = sqrt(var + ε)`, `x̂ = (x − μ) / σ`,
`y = γ ⊙ x̂ + β`.

**What was done**
- `update` stores `normalized` (x̂) and `standardDeviations` (σ) per token.
- `backpropagate(dY)`: stores `dY`; with `dX̂ = dY ⊙ γ`, returns per token

  ```
  dX = (1/σ) · (dX̂ − mean(dX̂) − x̂ · mean(dX̂ ⊙ x̂))
  ```

  The comment explains the three terms (the direct path, the path through
  the mean and the path through σ), and why each token's input errors sum
  to 0.
- `calculateGradients`: `dγ = Σ_t dY[t] ⊙ x̂[t]`, `dβ = Σ_t dY[t]`.
- Public fields: `scaleGradients`, `shiftGradients`, `gradientCount`.

**Check:** `LayerNormalization.test.ts` (3 tests): mean 0 and variance 1
with the initial scale and shift; a gradient check of `scale`, `shift` and
the input errors, with random scale and shift and inputs that have a
non-zero mean; each token's input errors sum to 0.

---

## Step 8: `FeedForward` backpropagation ✅ done

**Forward (per token):** `p = x·W1 + b1`, `a = GELU(p)`, `y = a·W2 + b2`.

**What was done**
- `gelu(x)` and the new `geluDerivative(x)` are exported module-level
  functions (as `relu` and `sigmoid` are in dnn). For
  `u = sqrt(2/π)(x + 0.044715x³)`:

  ```
  GELU′(x) = 0.5(1 + tanh u) + 0.5x(1 − tanh² u) · sqrt(2/π)(1 + 3·0.044715x²)
  ```

- `update` stores `inputs` (x), `preActivations` (p), `activations` (a).
- `backpropagate(dY)`: `dA = matrixTimesVector(W2, dY)`,
  `dP = dA ⊙ GELU′(p)`, returns `dX = matrixTimesVector(W1, dP)`; stores
  `dY` and `dP` (`preActivationErrors`).
- `calculateGradients`: `dW2 = Σ_t a ⊗ dY`, `db2 = Σ_t dY`,
  `dW1 = Σ_t x ⊗ dP`, `db1 = Σ_t dP`. The bias gradients stay empty when
  `bias` is false.
- Public fields: `firstLayerWeightGradients`, `firstLayerBiasGradients`,
  `secondLayerWeightGradients`, `secondLayerBiasGradients`, `gradientCount`.

**Check:** `FeedForward.test.ts` (4 tests): `geluDerivative` matches the
numerical slope of `gelu` from −4 to 4 to `1e-8`; basic `gelu` values; a
gradient check of all four parameter arrays and the input errors with
biases, and one without biases (whose bias gradients stay empty).

---

## Step 9: `GPTAttention` backpropagation (one head) ✅ done

**Forward:** `Q = XWq`, `K = XWk`, `V = XWv`; `S = QKᵀ / sqrt(d)`; causal
mask; `A = softmax(S)` row by row; `A′ = dropout(A)`; `C = A′V`.

**What was done**
- `update` stores `inputs` (X), `queries`, `keys`, `values`,
  `attentionWeights` (A) and `finalAttentionWeights` (A′); `attentionDropout`
  holds its mask.
- `backpropagate(dC)`, one commented block per step:
  1. Context vectors: `dA′[i][j] = dC[i] · V[j]`; `dV[j] = Σ_i A′[i][j] · dC[i]`.
  2. Dropout: `dA = attentionDropout.backpropagate(dA′)`.
  3. Softmax, row by row: `dS[i][j] = A[i][j] · (dA[i][j] − Σ_k A[i][k] · dA[i][k])`.
     The comment notes that masked positions have `A = 0`, so the causal
     mask needs no special handling.
  4. Scaling: `dS ← dS / sqrt(d)`.
  5. Scores: `dQ[i] = Σ_j dS[i][j] · K[j]`; `dK[j] = Σ_i dS[i][j] · Q[i]`.
  6. Inputs, summing the errors from the three projections:
     `dX = matrixTimesVector(Wq, dQ) + matrixTimesVector(Wk, dK) + matrixTimesVector(Wv, dV)`.
  - Stores `dQ`, `dK`, `dV`.
- `calculateGradients`: `dWq = Σ_t X ⊗ dQ` (likewise for K and V); biases
  when `qkvBias`.
- Public fields: `queryWeightGradients`, `keyWeightGradients`,
  `valueWeightGradients`, `queryBiasGradients`, `keyBiasGradients`,
  `valueBiasGradients`, `gradientCount`.

**Check:** `GPTAttention.test.ts` (5 tests): gradient checks with and
without biases; a gradient check with `dropoutRate = 0.5`, resetting the
seed before every forward pass so the same mask is drawn each time (the
test confirms some weights really were dropped); a **causality test**
(errors on token 0's context vector give exactly zero input errors for
tokens 1–3); rejection of errors for the wrong number of tokens.

---

## Step 10: `MultiHeadAttention` backpropagation ✅ done

**Forward:** concatenate the heads' `C` into `H` (one row per token), then
`Y = H·Wo + bo`.

**What was done**
- `update` stores `contextVectors` (H).
- `backpropagate(dY)`: `dH = matrixTimesVector(Wo, dY)`; splits each row of
  `dH` into one slice of `weightMatrixColumns` per head; passes each slice
  to its head's `backpropagate`; **sums** the heads' input errors.
- `calculateGradients`: `dWo = Σ_t H ⊗ dY`, `dbo = Σ_t dY`, then every
  head's `calculateGradients`.
- Public fields: `outputProjectionWeightGradients`,
  `outputProjectionBiasGradients`, `gradientCount`.

**Check:** `MultiHeadAttention.test.ts` (3 tests): a gradient check of
`Wo`, `bo`, head 0's query weights, head 1's key and value weights (checking
the errors are split between the heads correctly) and the input errors;
`calculateGradients` also fills every head's gradients; rejection of errors
for the wrong number of tokens.

---

## Step 11: `Transformer` backpropagation ✅ done

**Forward:**

```
s   = x + attentionOutputDropout(MHA(LN1(x)))
out = s + feedForwardOutputDropout(FF(LN2(s)))
```

**What was done**
- `backpropagate(dOut)`:

  ```
  dS = dOut + LN2.back(FF.back(feedForwardOutputDropout.back(dOut)))
  dX = dS   + LN1.back(MHA.back(attentionOutputDropout.back(dS)))
  ```

  using the existing `add` method for the sums. The comment explains why
  the unchanged copy of the errors along each shortcut stops the errors
  shrinking away in deep models.
- `calculateGradients`: delegates to `LN1`, `MHA`, `LN2` and `FF`.
- No stored state of its own: everything is kept by the sub-components.

**Check:** `Transformer.test.ts` (3 tests): a gradient check of one
parameter array from every trainable stage, plus the input errors; the same
check with both dropouts at 0.3 (reseeding before each forward pass); a
**shortcut test**: with the last layer of each branch set to zero (the
attention output projection and the feed forward second layer), the input
errors exactly equal the output errors.

---

## Step 12: `Embedding` backpropagation ✅ done

**Forward:** `e[t] = Tok[id_t] + Pos[t]`.

**What was done**
- The private matrices became `_textEmbeddingMatrix` and
  `_positionEmbeddingMatrix`, with public read-only getters
  `textEmbeddingMatrix` and `positionEmbeddingMatrix`. These are the names
  `WeightInitialization.test.ts` already used through a cast, so it still
  works.
- `getEmbeddingFromTokens` stores `tokenIds`.
- `backpropagate(dE)`: stores `dE` and returns nothing; the comment
  explains that token ids cannot be nudged, so backpropagation ends here.
- `calculateGradients`: `dTok[id_t] += dE[t]` and `dPos[t] += dE[t]`, adding
  into row views from `getRow`. The comment explains the repeated-token case
  and that the dense token gradient matrix is simple but wasteful.
- Public fields: `textEmbeddingGradients`, `positionEmbeddingGradients`,
  `gradientCount`.

**Check:** 4 new tests in `Embedding.test.ts`, with the token ids
`[5, 9, 5, 2]` (token 5 repeated):
- a gradient check of the rows of tokens 5, 9 and 2 (passed as row views,
  so nudging a row nudges the matrix) and of the whole position matrix;
- token 5's gradient is the sum of the errors at positions 0 and 2, and
  every other row of the 50,257 is exactly zero;
- unused positions get no gradient;
- errors for the wrong number of tokens are rejected.

**State after step 12:** 139 tests, all passing. Every component can now
backpropagate and calculate its gradients. Next: step 13 connects them
through `GPTModel`.

---

## Step 13: `GPTModel` backpropagation (whole model) ✅ done

**What was done**
- `backpropagate(outputErrors: number[][]): void` runs the chain in reverse:
  `linearOutputLayer → finalLayerNormalization → transformers (last to
  first) → embeddingDropout → embedding`. It takes the errors at the
  logits, not the targets, so the model contains no loss logic; the trainer
  will work out the errors with `crossEntropyErrors`. The doc comment notes
  that it must follow an `update()` in training or validation mode, since
  generation mode only returns the last token's logits.
- `calculateGradients(batchAverage)`: delegates to the embedding, every
  block, the final layer normalization and the output layer.

**Check:** 4 new tests in `src/test/GPTModel.test.ts`, on a two-block model
with the token ids `[464, 3290, 318, 3290]` and targets
`[3290, 318, 3290, 13]`:
- a gradient check of the **whole model with the real loss**: the mean
  cross entropy over the window. The helper's made-up loss is
  `output × R`, so the forward pass returns `[[meanLoss]]` and the output
  errors are `crossEntropyErrors(row, target, R / 4)`. Parameters are
  sampled from every kind of component:
  - token embedding rows of tokens in the window;
  - the position embeddings;
  - block 0's query weights and layer-norm scale;
  - block 1's value weights and feed forward weights and biases;
  - the final layer-norm shift;
  - output-layer rows for a target token and a non-target token;
- the same check with dropout 0.2 (reseeding before each forward pass);
- `calculateGradients` fills the gradients of every component (every
  `gradientCount` is 1);
- errors for the wrong number of tokens are rejected.

**Two settings the whole-model check needs**, with comments explaining them
in the test:
- **A smaller step, `1e-4`.** With the default `1e-2`, the token-embedding
  rows failed, some with numerical gradients of the opposite sign. The
  embeddings are tiny (standard deviation 0.02), and the first layer
  normalization magnifies changes in them about fifty times, so a step of
  half their size is not local. With `1e-4` every check passes, which shows
  the backpropagation was right and the step was the problem. This is worth
  showing students: numerical gradients need a step that is small compared
  with the values being nudged.
- **An absolute tolerance of `1e-9`.** Gradients of a mean loss can be
  around `1e-6`, which the default `1e-5` would pass whatever their value.

**Mutation check:** removing the final layer normalization from the
compiled backward pass makes both whole-model gradient checks fail.

**Still runs:** all 143 tests pass, and `node dist/index.js` still
generates text with the default (GPT-2 small) model in about 9 seconds. The
step 1 reference logits in generation mode are unchanged, so the output is
the same as before.

At this point the model can calculate correct gradients for every
parameter. Next: the optimizer (step 14).

---

## Step 14: Optimizer hyperparameters and `AdamW` ✅ done

**What was done**
- `Hyperparameters` has these values, as constructor parameters (arguments
  17–21) and getters/setters with validation, in the existing style:

  | Name | Default | Valid values | Notes |
  | --- | --- | --- | --- |
  | `learningRate` | 0.0004 | finite, > 0 | Book's chapter 5 value; dnn uses 0.001. |
  | `beta1` | 0.9 | ≥ 0 and < 1 | |
  | `beta2` | 0.999 | ≥ 0 and < 1 | |
  | `optimizerEpsilon` | 1e-8 | finite, > 0 | Named to avoid confusion with `LayerNormalization.epsilon`. |
  | `weightDecay` | 0.1 | finite, ≥ 0 | Book's value; dnn uses 0.01. 0 turns decay off. |

  They go **before** `seed`, which stays the last, optional argument. This
  moves `seed` from argument 17 to argument 22, but nothing passed it
  positionally (everything sets `hyperparameters.seed` instead). Three new
  validators: `validatePositiveNumber`, `validateBeta`,
  `validateWeightDecay`.
- New `src/AdamW.ts`, a port of dnn's, adapted to `Float32Array`. The class
  comment explains momentum, the per-parameter step size and decoupled
  weight decay for students:
  - `constructor(hyperparameters)`: the hyperparameters are read on **every
    update**, as in dnn, so changing the learning rate takes effect at once.
  - `step()`: increments the step count; called once per batch before any
    updates. `stepCount` getter.
  - `updateVector(parameters: Float32Array, gradients: ArrayLike<number>, applyWeightDecay = false)`,
    with `m` and `v` (Float32Arrays) kept in a `WeakMap<Float32Array, {m, v}>`
    keyed by the parameter array. The doc comment explains why the bias
    correction is needed:

    ```
    m = beta1 · m + (1 − beta1) · g
    v = beta2 · v + (1 − beta2) · g²
    mHat = m / (1 − beta1^t)
    vHat = v / (1 − beta2^t)
    p = p − learningRate · weightDecay · p        (if weight decay is applied)
    p = p − learningRate · mHat / (sqrt(vHat) + epsilon)
    ```

  - `updateMatrix(weights: Matrix, gradients: Matrix, applyWeightDecay = true)`:
    checks the shapes match, then calls
    `updateVector(weights.data, gradients.data, …)`. Because `Matrix` stores
    its values in one `Float32Array`, no row loop is needed.
  - Throws `Error` if `step()` has not been called, and `RangeError` if the
    gradients are the wrong size.
  - As in dnn, the step count `t` is shared by all arrays rather than kept
    per array (PyTorch keeps one per parameter). The two only differ for an
    array that is first updated after some steps have passed, which does not
    happen here: every array is updated at every step.

**Check**
- New `src/test/AdamW.test.ts` (10 tests):
  - updating before `step()` throws; `stepCount` counts the steps;
  - on the first step every parameter moves by `learningRate` against its
    gradient, for gradients from 0.001 to 200 of both signs;
  - a zero gradient leaves a parameter unchanged without weight decay and
    multiplies it by `1 − learningRate · weightDecay` with it;
  - three steps with changing gradients match a hand calculation of the
    formula in 64-bit, to 1e-7;
  - momentum: after ten steps one way, one reversed gradient does not
    reverse the parameter's direction;
  - two arrays with opposite constant gradients each move by exactly
    `learningRate` per step (so their moments are separate);
  - `updateMatrix` updates every value, with weight decay by default;
  - a change to `learningRate` takes effect at the next update;
  - gradients of the wrong size are rejected.
- 4 new tests in `src/test/Hyperparameters.test.ts`: the defaults, the
  setters storing valid values, the setters rejecting invalid ones without
  changing state, and the constructor rejecting invalid values.

**State after step 14:** 157 tests, all passing. The optimizer exists but is
not connected to the model yet; that is step 15.

---

## Step 15: `updateWeights` in every component

**Changes**
- Every component with parameters gets `updateWeights(optimizer)`:
  - **weight decay on**: all weight matrices and both embedding matrices;
  - **weight decay off**: all biases and layer-norm scale and shift (as in
    dnn and GPT-2 practice; the book decays everything, so add a comment).
- `GPTModel.updateWeights(optimizer)` calls `optimizer.step()` once, then
  every component's `updateWeights`, as dnn's `MNISTNetwork` does.

**Check**
- One-step test on a tiny model: run forward, backpropagate,
  `calculateGradients(false)`, `updateWeights` on one window, then measure
  the loss on the **same** window in `VALIDATION_MODE` before and after. The
  loss must decrease.
- Every parameter array in the model changes after the update. (Even
  token-embedding rows for tokens not in the window change, because weight
  decay shrinks them.)

---

## Step 16: The training loop in `Trainer`

**Changes**
- `train()` becomes the dnn loop, per window instead of per image:

  ```
  optimizer = new AdamW(hyperparameters)          // created once, so its moments persist across epochs
  for each epoch:
    shuffle windows; create batches
    for each batch:
      for each window (index w) in the batch:
        logits = model.update(window.inputs)                                    // forward
        loss  += Σ_t crossEntropyLoss(logits[t], window.targets[t])
        errors = logits.map((row, t) => crossEntropyErrors(row, window.targets[t], 1 / contextLength))
        model.backpropagate(errors)                                             // errors flow backwards
        model.calculateGradients(w !== 0)                                       // first window replaces, the rest average
      model.updateWeights(optimizer)                                            // one AdamW step per batch
    validation loss (unchanged: forward only, VALIDATION_MODE)
  ```

- Remove the "Weights are not updated yet" comments; update the
  `numberEpochs` documentation in `Hyperparameters` (more epochs now give
  different losses).

**Check**
- Existing `Trainer` tests still pass (mode restoration, shuffling, window
  counts).
- New **overfitting test**: a tiny model trained for ~10 epochs on a short,
  repetitive text gives a final training loss well below the first epoch's
  and below `ln(vocabularySize)` ≈ 10.8.
- Validation loss changes between epochs (previously identical).

---

## Step 17: Real training run

**Changes**
- `index.ts`: uncomment the trainer, use the demo configuration in C2, train
  on `data/the-verdict.txt`, then generate text with the trained model.

**Check**
- The training loss falls steadily over the epochs.
- The validation loss falls at first; with this small text it will probably
  start rising again later (overfitting), as in the book.
- The generated text after training looks more like English than before.

---

## Step 18: Documentation

- Update `README.md` with a short description of how training works and how
  to run it.
- Make sure every new method has an API-style doc comment, per `CLAUDE.md`.

---

# Part C: Reference

## C1. How dnn does it

| dnn | Role | LLM equivalent |
| --- | --- | --- |
| `HiddenLayer.update(inputs)` | Forward pass; stores `_inputs`, `_activations`. | `update(inputs)` in each component |
| `HiddenLayer.calculateErrors` + `calculatePreviousLayerErrors` | Errors before the activation; errors at the inputs. | `backpropagate(outputErrors)` |
| `HiddenLayer.calculateGradients(batchAverage)` | Error × input; `false` replaces, `true` running mean. | Same |
| `HiddenLayer.updateWeights` / `updateBiases` | AdamW; decay on weights only. | `updateWeights(optimizer)` |
| `OutputLayer.calculateErrors(target)` | `probability − oneHot`. | `Loss.crossEntropyErrors` |
| `Loss.crossEntropyLoss` | `−log(probability of target)`. | `Loss.crossEntropyLoss` |
| `MNISTNetwork.backpropagate / calculateGradients / updateWeights` | Delegates to each layer; `step()` first. | `GPTModel` |
| `AdamW` | `step()`, `updateMatrix`, `updateVector`, moments in a `WeakMap`. | `AdamW` (Float32Array) |
| `MNISTTrainer.train()` | Per example: forward, loss, backpropagate, `calculateGradients(!isStartOfBatch)`; per batch: `updateWeights`. | `Trainer.train()` |

## C2. Demo configuration for training runs

The default (GPT-2 small, ~162M parameters) configuration is far too large to
train in plain TypeScript: AdamW needs a gradient and two moments per
parameter, about 2.6 GB in total, and each step would be very slow.

| Hyperparameter | Value |
| --- | --- |
| `embeddingSize` | 64 |
| `numberAttentionHeads` | 4 |
| `numberTransformerBlocks` | 2 |
| `contextLength` | 32 |
| `batchSize` | 4 |
| `numberEpochs` | 10 |

The token embedding and output layer alone are still 2 × 50,257 × 64 ≈ 6.4
million parameters, so the vocabulary dominates. A smaller vocabulary is a
possible later change, but out of scope here.

## C3. Optional extras (after step 18)

- **Gradient clipping** by global norm (book appendix D), with a
  `gradientClipNorm` hyperparameter, off by default.
- **Learning-rate warm-up and cosine decay** (book appendix D).
- **Saving and loading weights**, so a trained model can be reused without
  retraining.
- **Evaluating every N batches** and printing a generated sample after each
  epoch, as in the book's `train_model_simple`.
- **Weight tying** between the token embedding and the output layer, as in
  GPT-2: the gradients from both uses are summed into one matrix (a good
  fan-out example, but it changes the parameter count from the book's).
