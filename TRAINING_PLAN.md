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
- Batch averaging copies dnn exactly. Each component keeps a
  `_gradientCount`:
  - `calculateGradients(false)` (first window of a batch) replaces the stored
    gradients and sets the count to 1;
  - `calculateGradients(true)` (remaining windows) increments the count to `k`
    and folds the new gradients into a running mean:
    `mean += (new − mean) / k`.
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

## Step 6: `LinearOutputLayer` backpropagation

**Forward:** `z[t] = W · h[t]` (W is vocabSize × embeddingSize, no bias).

**Changes**
- `update` stores `inputs` (h).
- `backpropagate(dZ)`: stores `dZ`; returns `dH[t] = vectorTimesMatrix(dZ[t], W)`.
- `calculateGradients(batchAverage)`: `dW = Σ_t dZ[t] ⊗ h[t]`, with
  replace/running mean as in A3.
- Getters: `weightGradients`, `gradientCount`.

**Check**
- `checkGradients` on a tiny layer (use a small `embeddingSize`; the vocabulary
  is large, so sample parameter indices rather than checking them all) for
  both `W` and the input errors.
- A **batch-averaging test**: `calculateGradients(false)` after window A then
  `calculateGradients(true)` after window B gives the mean of the gradients
  of A and B computed separately. (This tests the A3 convention once; later
  components use the same code pattern.)

---

## Step 7: `LayerNormalization` backpropagation

**Forward (per token):** `μ`, `σ = sqrt(var + ε)`, `x̂ = (x − μ) / σ`,
`y = γ ⊙ x̂ + β`.

**Changes**
- `update` stores `normalized` (x̂) and `standardDeviations` (σ) per token.
- `backpropagate(dY)`: stores `dY`; with `dX̂ = dY ⊙ γ`, returns per token

  ```
  dX = (1/σ) · (dX̂ − mean(dX̂) − x̂ · mean(dX̂ ⊙ x̂))
  ```

  The comment explains the three terms: the direct path, the path through
  the mean, and the path through the variance.
- `calculateGradients`: `dγ = Σ_t dY[t] ⊙ x̂[t]`, `dβ = Σ_t dY[t]`.

**Check**
- `checkGradients` for `scale`, `shift` and the input errors. Set `scale` and
  `shift` to random values first, so the test does not only cover `γ = 1`,
  `β = 0`.
- Step 1's reference logits are unchanged.

---

## Step 8: `FeedForward` backpropagation

**Forward (per token):** `p = x·W1 + b1`, `a = GELU(p)`, `y = a·W2 + b2`.

**Changes**
- Move `gelu` to a module-level exported function and add
  `geluDerivative(x)`. For `u = sqrt(2/π)(x + 0.044715x³)`:

  ```
  GELU′(x) = 0.5(1 + tanh u) + 0.5x(1 − tanh² u) · sqrt(2/π)(1 + 3·0.044715x²)
  ```

- `update` stores `inputs` (x), `preActivations` (p), `activations` (a).
- `backpropagate(dY)`:
  - `dA[t] = matrixTimesVector(W2, dY[t])`
  - `dP[t] = dA[t] ⊙ GELU′(p[t])`
  - returns `dX[t] = matrixTimesVector(W1, dP[t])`
  - stores `dY` and `dP`.
- `calculateGradients`: `dW2 = Σ_t a[t] ⊗ dY[t]`, `db2 = Σ_t dY[t]`,
  `dW1 = Σ_t x[t] ⊗ dP[t]`, `db1 = Σ_t dP[t]` (biases only when `bias` is true).

**Check**
- `geluDerivative` matches `(gelu(x+h) − gelu(x−h)) / 2h` for x in
  `[−4, 4]`.
- `checkGradients` for all four parameter arrays and the input errors, with
  `feedForwardBias` both true and false.

---

## Step 9: `GPTAttention` backpropagation (one head)

**Forward:** `Q = XWq`, `K = XWk`, `V = XWv`; `S = QKᵀ / sqrt(d)`; causal
mask; `A = softmax(S)` row by row; `A′ = dropout(A)`; `C = A′V`.

**Changes**
- `update` stores `inputs` (X), `queries`, `keys`, `values`,
  `attentionWeights` (A) and `finalAttentionWeights` (A′);
  `attentionDropout` already holds its mask.
- `backpropagate(dC)`, one commented block per step:
  1. Context vectors: `dA′[i][j] = dC[i] · V[j]`; `dV[j] = Σ_i A′[i][j] · dC[i]`.
  2. Dropout: `dA = attentionDropout.backpropagate(dA′)`.
  3. Softmax, row by row: `dS[i][j] = A[i][j] · (dA[i][j] − Σ_k A[i][k] · dA[i][k])`.
     Masked positions have `A = 0`, so their errors are automatically 0: the
     causal mask needs no special handling (worth a comment).
  4. Scaling: `dS ← dS / sqrt(d)`.
  5. Scores: `dQ[i] = Σ_j dS[i][j] · K[j]`; `dK[j] = Σ_i dS[i][j] · Q[i]`.
  6. Inputs (fan-out to three projections, so the errors are summed):
     `dX[t] = matrixTimesVector(Wq, dQ[t]) + matrixTimesVector(Wk, dK[t]) + matrixTimesVector(Wv, dV[t])`.
  - Stores `dQ`, `dK`, `dV`.
- `calculateGradients`: `dWq = Σ_t X[t] ⊗ dQ[t]` (likewise for K and V);
  biases when `qkvBias`.

**Check**
- `checkGradients` for `Wq`, `Wk`, `Wv` and the input errors, with
  `qkvBias` true and false, and `dropoutRate = 0`.
- A dropout version: `dropoutRate = 0.5`. The helper reruns `forward()`, which
  would draw a new mask, so for this test reset `hyperparameters.seed` before
  every forward pass so the same mask is drawn each time.
- A causality test: errors placed only on the context vector of token 0 give
  zero input errors for tokens 1..n−1.
- Step 1's reference logits are unchanged.

---

## Step 10: `MultiHeadAttention` backpropagation

**Forward:** concatenate the heads' `C` into `H` (one row per token), then
`Y = H·Wo + bo`.

**Changes**
- `update` stores `contextVectors` (H).
- `backpropagate(dY)`:
  - `dH[t] = matrixTimesVector(Wo, dY[t])`;
  - split each row of `dH` into one slice of `weightMatrixColumns` per head
    (the reverse of the concatenation);
  - pass each slice to its head's `backpropagate`;
  - **sum** the heads' returned input errors, because every head received
    the same embeddings (fan-out).
- `calculateGradients`: `dWo = Σ_t H[t] ⊗ dY[t]`, `dbo = Σ_t dY[t]`; then
  each head's `calculateGradients`.

**Check**
- `checkGradients` for `Wo`, `bo`, one head's `Wq`, and the input errors,
  with 2 heads.

---

## Step 11: `Transformer` backpropagation

**Forward:**

```
s   = x + dropout1(MHA(LN1(x)))
out = s + dropout2(FF(LN2(s)))
```

**Changes**
- `backpropagate(dOut)`:

  ```
  dS = dOut + LN2.back(FF.back(dropout2.back(dOut)))
  dX = dS   + LN1.back(MHA.back(dropout1.back(dS)))
  ```

  where `dropout1` is `attentionOutputDropout` and `dropout2` is
  `feedForwardOutputDropout`. The comment stresses that each shortcut copies
  the errors straight through, unchanged, and they are added to the errors
  from the branch. This is why residual connections stop gradients
  vanishing in deep models.
- `calculateGradients`: delegates to `MHA`, `FF`, `LN1`, `LN2`.
- Reuse the existing `add` method for the error sums.

**Check**
- `checkGradients` for one parameter array from each of the four
  sub-components, and the input errors.
- A shortcut test: with every weight in `MHA` and `FF` set to 0, the input
  errors equal the output errors exactly.

---

## Step 12: `Embedding` backpropagation

**Forward:** `e[t] = Tok[id_t] + Pos[t]`.

**Changes**
- `getEmbeddingFromTokens` stores `tokenIds`.
- `backpropagate(dE)`: stores `dE`. Returns nothing: token ids are not
  differentiable, so backpropagation stops here (worth a comment).
- `calculateGradients`:
  - `dTok[id_t] += dE[t]` for every position: only rows of tokens in the
    window are non-zero, and a token that appears twice receives both
    contributions;
  - `dPos[t] += dE[t]`.
- `dTok` is a dense full-size matrix, for clarity. A comment can mention that
  real implementations treat it as sparse.
- `textEmbeddingMatrix` and `positionEmbeddingMatrix` are private. Add
  read-only getters for them (and their gradients), so the gradient check
  can nudge their values and `updateWeights` can be tested.

**Check**
- `checkGradients` with a token sequence that **repeats a token id**,
  sampling rows that appear in the window, and rows that do not (whose
  gradient must be exactly 0).

---

## Step 13: `GPTModel` backpropagation (whole model)

**Changes**
- `backpropagate(outputErrors: number[][]): void` runs the chain in reverse:
  `linearOutputLayer → finalLayerNormalization → transformers (last to
  first) → embeddingDropout → embedding`.
- `calculateGradients(batchAverage)`: delegates to every component.
- `backpropagate` takes the output errors, not the targets, so the model
  contains no loss logic and the trainer shows where the errors come from.

**Check**
- `checkGradients` on the whole tiny model (2 blocks) with the **real loss**:
  the forward function returns the logits, and the loss is
  `Σ_t crossEntropyLoss(logits[t], target[t]) / contextLength`, with output
  errors from `crossEntropyErrors(…, 1 / contextLength)`. Sample parameters
  from the embedding, the first and last block, the final layer norm and the
  output layer.
- Step 1's reference logits are unchanged.

At this point the model can calculate correct gradients for every parameter.

---

## Step 14: Optimizer hyperparameters and `AdamW`

**Changes**
- `Hyperparameters` gets these values, with validation and setters in the
  existing style:

  | Name | Default | Notes |
  | --- | --- | --- |
  | `learningRate` | 0.0004 | Book's chapter 5 value; dnn uses 0.001. |
  | `beta1` | 0.9 | |
  | `beta2` | 0.999 | |
  | `optimizerEpsilon` | 1e-8 | Named to avoid confusion with `LayerNormalization.epsilon`. |
  | `weightDecay` | 0.1 | Book's value; dnn uses 0.01. |

- New `src/AdamW.ts`, a port of dnn's, adapted to `Float32Array`:
  - `step()`: increments `t`; called once per batch before any updates.
  - `updateVector(parameters: Float32Array, gradients: Float32Array, applyWeightDecay: boolean)`,
    with `m` and `v` kept in a `WeakMap<Float32Array, {m, v}>`:

    ```
    m = beta1 · m + (1 − beta1) · g
    v = beta2 · v + (1 − beta2) · g²
    mHat = m / (1 − beta1^t)
    vHat = v / (1 − beta2^t)
    p = p − learningRate · weightDecay · p        (if weight decay is applied)
    p = p − learningRate · mHat / (sqrt(vHat) + epsilon)
    ```

  - `updateMatrix(weights: Matrix, gradients: Matrix, applyWeightDecay = true)`:
    calls `updateVector(weights.data, gradients.data, …)`. Because `Matrix`
    stores its values in one `Float32Array`, no row loop is needed.
  - Throws if `step()` has not been called.

**Check**
- New `src/test/AdamW.test.ts`:
  - after the first step, each parameter moves by ≈ `learningRate` against the
    sign of its gradient (bias correction makes `mHat / sqrt(vHat) = ±1`);
  - weight decay shrinks parameters with a zero gradient only when applied;
  - two different arrays keep separate moments;
  - calling an update before `step()` throws.

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
