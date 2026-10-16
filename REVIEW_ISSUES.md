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

## Phase 3: Training loop

### [ ] 8. Bring the training loop into line with the book

**Where:** [src/Trainer.ts:68-179](src/Trainer.ts#L68-L179)

**Problem:**
- Loss is averaged per window, then the window averages are averaged again. The book averages over every token in the batch. The results differ only when the last window is shorter.
- The last incomplete window is kept. The book drops it (`drop_last=True`).
- There is no train/validation split. The book uses 90/10.
- Batches are not shuffled.

**Fix:**
- Split the tokens 90/10 into training and validation sets.
- Drop the incomplete final window.
- Shuffle the order of the windows on each epoch.
- Average loss and gradients over every token in a batch of `batchSize` windows, then do one weight update per batch.
- Calculate the validation loss with `hyperparameters.mode = Hyperparameters.VALIDATION_MODE`. In this mode dropout is off and the model returns logits for every position, so `calculateLoss()` can be reused unchanged.

**Done when:** each epoch reports training and validation loss, and the weights are updated once per batch.

---

### [ ] 9. Generate from token ids instead of re-tokenizing text

**Where:** `Generator.generate()` in [src/Generator.ts](src/Generator.ts)

**Problem:** Each step decodes one token to text, appends it, then re-encodes the whole string. BPE doesn't always give back the same token ids, because new text can merge with the previous token.

**Fix:**
- Encode the prompt once.
- Keep a `number[]` of token ids and append each chosen id to it.
- Feed the last `contextLength` ids to the model at each step.
- Decode once at the end.
- This means `Decoder.greedy` should return a token id rather than text.

**Done when:** generation never re-encodes text that has already been encoded.

---

## Phase 4: Small fixes and tidy-ups

### [ ] 10. Fix stale comment
- [src/Embedding.ts:69-75](src/Embedding.ts#L69-L75) says Hugging Face uses 0.01 for position embeddings. Hugging Face uses 0.02; the 0.01 value comes from OpenAI's original TensorFlow code.

### [ ] 11. Check that token ids are in range
- [src/Embedding.ts:152](src/Embedding.ts#L152): an id outside `0 … vocabSize - 1` returns `undefined`, which later fails with a confusing `TypeError`. Throw a clear `RangeError` instead.

### [ ] 12. Allow `<|endoftext|>` to be encoded
- [src/BPETokenizer.ts:17-19](src/BPETokenizer.ts#L17-L19): `encode` throws "Disallowed special token found" for text containing `<|endoftext|>`.
- The book passes `allowed_special={"<|endoftext|>"}`, which you'll need when joining several documents together for training.

### [ ] 13. Remove the unused path in `index.ts`
- [src/index.ts:6](src/index.ts#L6): `filePath` is computed but never used, and `trainer.load()` is given a relative path. Use `filePath`, or remove it.

### [ ] 14. Validate `contextLength`
- The `contextLength` setter in [src/Hyperparameters.ts](src/Hyperparameters.ts): the setter and the constructor accept any value. Use `validatePositiveInteger`, like the other numeric hyperparameters.

### [ ] 15. Note on `contextLength` and the parameter count
- The default `contextLength` of 12 (GPT-2 uses 1024) means the parameter count only matches the book's 163,009,536 when it is set to 1024. Mention this in a comment or test so it doesn't look like a regression.

---

## Later: loading OpenAI's GPT-2 weights (optional)

If you later want to load OpenAI's pretrained weights:
- [ ] Set `qkvBias = true`, because real GPT-2 has biases on its query/key/value projections.
- [ ] Split OpenAI's combined `c_attn` weight matrix into separate query, key and value weights for each head.
- [ ] Tie the output layer's weights to the token embedding. This reduces the parameter count to about 124M.
- [ ] Note that OpenAI stores linear weights as `[in, out]`, which matches this code's `embeddingSize × columns` layout.
