# GPT-2 Implementation Review: Issues to Fix

A review of the forward pass against GPT-2 and *Build a Large Language Model (from Scratch)*.

The forward-pass maths is correct. The issues below are structural differences from the book, design problems that will block training, and tidy-ups. They are listed in the recommended order of work.

Tick each box when the issue is done.

---

## Things already confirmed correct

These need no work. They are listed here for reference.

- Tokenizer: `r50k_base` has the same 50,257-token vocabulary as GPT-2.
- Token and position embeddings are added together, then dropout is applied.
- Transformer blocks are pre-norm: layer norm comes before attention and before the feed-forward network. Dropout is applied to the output of each before it is added to the shortcut.
- Layer normalisation uses the population variance, ε = 1e-5, scale initialised to 1 and shift to 0.
- Attention scores are scaled by √(head size), future positions are masked with `-Infinity` before a single softmax, and dropout is applied to the attention weights.
- Multi-head attention uses separate weights per head, concatenates the heads, then applies an output projection with bias. This is equivalent to the book's version, which splits one large matrix.
- Feed-forward network: 768 → 3072, GELU (tanh approximation), → 768, with bias.
- Final layer norm, then an output layer with no bias and no weight tying (the book's `out_head`).
- Parameter count is 163,009,536 when `contextLength` = 1024, which matches the book.
- Loss: a numerically stable log-softmax, with each target being the input shifted by one token.
- Weights, biases and layer norm values are stored in `Float32Array`s (32-bit, as PyTorch uses by default), through the `Matrix` class.
- Weight initialisation follows GPT-2: N(0, 0.02) for every linear layer and the token embedding, N(0, 0.01) for the position embedding, both residual projections scaled by 1/√(2 · blocks), biases 0, layer norm scale 1 and shift 0.

---

## Phase 2: Prepare the structure for backpropagation

### [ ] 5. Keep the values needed by the backward pass

**Where:** every layer: `Embedding`, `GPTAttention`, `MultiHeadAttention`, `LayerNormalization`, `FeedForward`, `LinearOutputLayer`, `Transformer`, `GPTModel`

**Problem:** None of the intermediate values from the forward pass are kept. The backward pass will need:

| Layer | Values to keep |
|---|---|
| `Embedding` | Token ids |
| `GPTAttention` | Input, queries, keys, values, attention weights (after softmax), dropout mask |
| `MultiHeadAttention` | Concatenated context vectors |
| `LayerNormalization` | Normalised values, standard deviation (or 1/std) per token |
| `FeedForward` | Input, values before GELU, values after GELU |
| `LinearOutputLayer` | Input |
| `Transformer` / `GPTModel` | Dropout masks, shortcut inputs |

Dropout masks matter most, because they are random draws and can't be recalculated.

**Fix:**
- Give each layer a `forward()` that stores what it needs.
- Give each layer a `backward(gradOutput)` that returns the gradient for its input and adds up gradients for its own weights.
- Add a gradient `Matrix` (or `Float32Array` for biases and layer norm values) with the same shape as each weight, and a `zeroGradients()` method.
- The backward pass of a linear layer needs two more products besides `vectorTimesMatrix`: the gradient times the transposed weights (for the input's gradient), and the outer product of the input and the gradient (for the weights' gradient). Add them to [src/Matrix.ts](src/Matrix.ts) next to the existing helpers.

**Done when:** each layer can do a forward pass followed by a backward pass, and a finite-difference gradient check on a tiny model (for example `embeddingSize` 8, 2 heads, 1 block) agrees with the backward pass.

---

### [ ] 6. Performance

This model is a learning aid, so **clarity comes before speed**. Each item below says whether it's worth doing on those terms.

#### Done: weights stored in `Float32Array`

- [src/Matrix.ts](src/Matrix.ts) holds each weight matrix as one `Float32Array`, stored row by row, with `get`, `set`, `getRow` and `toArrays` methods.
- Linear layers use two documented helpers: `vectorTimesMatrix` (input × weights + biases) and `matrixTimesVector` (for the output layer, which has one row per vocabulary token).
- Biases and layer norm scale and shift are `Float32Array`s too.
- Activations (embeddings, queries, context vectors, logits and so on) are still plain `number[]` and `number[][]` arrays, so the forward pass stays easy to read and print.

Measured on the full-size model (162M parameters, `contextLength` 12):

| | Before (`number[][]`) | After (`Float32Array`) |
|---|---|---|
| Build time | 18.5 s | 7.4 s |
| Memory for the model | 1.34 GB | 0.66 GB |
| One forward pass of 12 tokens | 26.9 s | 1.7 s |

The logits differ by at most 1.6e-7, from rounding the stored weights to 32 bits. Most of the 15× speed-up comes from reading the weights in storage order, row by row. The old code went down each column of a `number[][]`, jumping between separate arrays.

#### Remaining performance problems

| # | Problem | Effect | Possible fix | Worth it for a learning aid? |
|---|---|---|---|---|
| 6a | **Full-size training will be slow.** A forward pass of one 12-token window takes about 1.7 s, and a backward pass usually takes about twice as long as a forward pass. `the-verdict.txt` gives about 396 training windows. | About 35 minutes per epoch, so about 6 hours for the book's 10 epochs. | Give students a small configuration, for example `embeddingSize` 256, 4 heads and 4 blocks (about 16M parameters and about 8× fewer multiply-adds per token). Keep 768/12/12 for checking the parameter count against the book. | **Yes.** It needs no code changes, just a documented preset. |
| 6b | **The output layer dominates.** Each position needs 50,257 × 768 ≈ 38.6M multiply-adds, against about 85M for all 12 transformer blocks together. Training needs logits for every position, then a 50,257-term softmax for each one. | About a third of the forward pass, and the same share of the backward pass. | Unavoidable with a full-size vocabulary. A smaller `embeddingSize` (6a) reduces it in proportion. | No code change needed. |
| 6c | **Each token is multiplied separately.** Every linear layer is a vector × matrix product per token, so each weight matrix is read from memory once per token. Libraries such as PyTorch multiply all the tokens in a batch at once (matrix × matrix), splitting the work into blocks that fit in the CPU cache. | Probably the largest remaining speed-up, perhaps 2–5×. | A `matrixTimesMatrix` helper that works through the tokens in blocks. | **Probably not.** Working one token at a time matches how the maths is usually explained. Possible later as an optional "fast path" with the simple version kept as the reference. |
| 6d | **Activations are built with `map`, `reduce` and `Array.from`.** These create many short-lived arrays and closures, which the garbage collector then has to clear up. | A moderate slowdown, and some memory churn. | Plain `for` loops writing into reusable `Float32Array`s. | **No.** `map` and `reduce` are easier to follow, and the weights matter far more. |
| 6e | **Only one CPU core is used.** JavaScript runs on one thread, so the windows in a batch are processed one after another. | Other cores sit idle. | Worker threads, each running some of the batch's windows. Because the weights are now `Float32Array`s, they can live in a `SharedArrayBuffer` and be shared between workers instead of copied. Gradients would need adding together across workers. | **Maybe later**, as an optional extension once training works. It adds concurrency, which is a separate topic. |
| 6f | **No GPU or SIMD.** Plain JavaScript can't use the CPU's vector instructions or a GPU. | Real frameworks are 100× or more faster. | WebGPU, TensorFlow.js or ONNX Runtime. | **No.** These hide the arithmetic that the project is meant to show. |
| 6g | **Generation recomputes everything.** Each new token reruns the whole context through the model. Real implementations keep a "KV cache" of earlier tokens' keys and values. | Small with `contextLength` 12. With 1024, each token costs as much as the whole context. | A KV cache in `GPTAttention`, used only in `GENERATION_MODE`. | **Optional.** It's a well-known technique and worth explaining, but it makes `GPTAttention` more complicated. |
| 6h | **Attention scores are calculated for masked positions too.** The full tokens × tokens score matrix is calculated, then the upper half is set to `-Infinity`. | Negligible at `contextLength` 12. | Calculate scores only for column ≤ row. | **No.** The full matrix and then the mask match the book's explanation. |
| 6i | **Memory for training.** Gradients need a second copy of every weight, and AdamW needs two more (its first and second moment estimates). The backward pass also needs each window's activations kept (issue 5). | About 0.66 GB × 4 ≈ 2.6 GB for the full model, plus activations. | Already as small as it can be without lower-precision tricks. Node needs `--max-old-space-size` raised for the full model. A smaller configuration (6a) avoids the problem. | Document the `node` flag. |
| 6j | **Verbose debug output.** With `DEBUG.VERBOSE` on, `console.table` prints whole matrices, including 50,257-column logit rows, which can take minutes. | Very slow runs when verbose. | Print only a slice, for example the first 10 rows and columns, when a matrix is large. | **Yes.** It's small and makes the debug output more useful. |

**Done when:** the items marked "Yes" are done, and the others are either done as optional extensions or recorded as deliberate choices.

---

## Phase 3: Training

### [ ] 8. Update the weights once per batch

**Where:** `Trainer.train()` in [src/Trainer.ts](src/Trainer.ts), at the comment "Once backpropagation has been written…"

**Needs:** issue 5 (the backward pass).

**Problem:** The training loop already follows the book: a 90/10 split, full windows only, shuffled batches with the incomplete last batch dropped, loss averaged over every token in a batch, and validation loss in `VALIDATION_MODE`. But the weights are never updated, so every epoch gives the same validation loss.

**Fix:**
- After `calculateBatchLoss()`, zero the gradients, run the backward pass from the batch loss, and update the weights once per batch.
- The gradient of the batch loss must be averaged over every token in the batch, in the same way as the loss.
- Add an optimiser. The book uses AdamW with a learning rate of 0.0004 and weight decay of 0.1, so add those as hyperparameters.
- Once the weights change, raise the default `numberEpochs` from 1 to the book's 10.

**Done when:** the training and validation losses fall over several epochs on `the-verdict.txt`.

---

## Later: loading OpenAI's GPT-2 weights (optional)

If you later want to load OpenAI's pretrained weights:
- [ ] Set `qkvBias = true`, because real GPT-2 has biases on its query/key/value projections.
- [ ] Split OpenAI's combined `c_attn` weight matrix into separate query, key and value weights for each head.
- [ ] Tie the output layer's weights to the token embedding. This reduces the parameter count to about 124M.
- [ ] Note that OpenAI stores linear weights as `[in, out]`, which matches this code's `embeddingSize × columns` layout.
