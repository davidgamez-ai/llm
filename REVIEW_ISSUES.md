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
- Add a gradient array for each weight array, and a `zeroGradients()` method.

**Done when:** each layer can do a forward pass followed by a backward pass, and a finite-difference gradient check on a tiny model (for example `embeddingSize` 8, 2 heads, 1 block) agrees with the backward pass.

---

### [ ] 6. Store weights in flat typed arrays

**Where:** every weight matrix

**Problem:** The full model has 163M parameters. As `number[][]` they are slow, and they use a lot of memory, especially once the gradient arrays and optimiser state are added.

**Fix:**
- Store each matrix as a `Float32Array` with known rows and columns, indexed as `row * columns + column`.
- Write shared helpers for multiplying a matrix by a vector or matrix, and for multiplying by a transposed matrix. The backward pass needs both.

**Done when:** the forward pass gives the same results as before, within floating-point tolerance, and runs faster.

---

### [ ] 7. Choose one weight initialisation scheme

**Where:** [src/GPTAttention.ts:88-117](src/GPTAttention.ts#L88-L117), [src/FeedForward.ts:73-96](src/FeedForward.ts#L73-L96), [src/Embedding.ts:82-103](src/Embedding.ts#L82-L103), [src/MultiHeadAttention.ts:70-95](src/MultiHeadAttention.ts#L70-L95), [src/LinearOutputLayer.ts:66-89](src/LinearOutputLayer.ts#L66-L89)

**Problem:** The code mixes initialisation styles:
- Embeddings, the output layer and the attention output projection use GPT-2's normal-distribution initialisation.
- The query/key/value weights and the feed-forward layers use PyTorch's default `nn.Linear` initialisation.
- The book uses PyTorch defaults everywhere, including N(0,1) for embeddings.

**Fix:** Choose one scheme. For GPT-2's scheme:
- Every linear layer uses N(0, 0.02), with biases set to 0.
- The token embedding uses N(0, 0.02). The position embedding uses N(0, 0.02) to match Hugging Face, or N(0, 0.01) to match OpenAI's original code.
- **Both** residual projections are scaled by 1/√(2 · numberTransformerBlocks): the attention output projection (already done) and the **second feed-forward layer** (currently missing).
- Consider one shared `randomNormal` helper. It is currently copied into three classes.

**Done when:** every layer follows the chosen scheme, and the comments say which scheme it is.

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
