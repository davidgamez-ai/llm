# Plan: training on any of the four datasets

This plan changes `Trainer` so it can train on The Verdict, Tiny Shakespeare,
TinyStories or Alpaca, using one training loop. It is written as a sequence of
small steps. Each step leaves the code compiling and all tests passing
(`npm test`), and ends with a check.

## Progress

| Step | Status |
|---|---|
| 1. Fix the tokenizer crash | Done |
| 2. Weight the batch correctly | Done |
| 3. `Dataset`, `TextDataset`, `Trainer` on windows | Done |
| 4. `TinyStoriesDataset` | Done |
| 5. Rename the story hyperparameters | Done |
| 6. `trainingWindowLength` | Done |
| 7. `AlpacaDataset` | Done |
| 8. A real run on each dataset | Done; training loss recorded for The Verdict only, and the rerun for the others was skipped by decision |
| 9. Documentation | Done |

`TextDataset` was added after steps 4 and 5, but the rest of step 3 was done with
step 4, which needed it.

---

# Part A: The position when this plan was written

A1 to A3 describe the code before step 1. See Progress above for what has changed.

## A1. What `Trainer` did

`Trainer` mixes two jobs: preparing data and running the training loop. It has
two loaders, `load()` for text files and `loadTinyStories()` for the CSV files.
Both produce one long list of training tokens and one of validation tokens, and
`createWindows()` cuts these into windows of `contextLength` tokens when
`train()` runs.

The training loop itself is already general. `trainOnBatch()` runs each window
through the model separately, so windows do not need padding and can differ in
length.

## A2. The four datasets

| Dataset | File | Form | Validation data | Works now? |
|---|---|---|---|---|
| The Verdict | `data/the-verdict.txt` | One continuous text | Last 10% of the text | Yes; on the large model it needs `trainingWindowLength` |
| Tiny Shakespeare | `data/tinyshakespeare.txt` | One continuous text | Last 10% of the text | Yes, since step 1; not yet run through training (step 8) |
| TinyStories | `data/TinyStories_*.csv` | Separate stories | Its own file | Yes, through `TinyStoriesDataset` |
| Alpaca | `data/alpaca_data.json` | Separate examples | None: held out by `AlpacaDataset` | Yes, through `AlpacaDataset`; only a very short run so far (step 8) |

## A3. Problems found while preparing this plan

**1. `BPETokenizer.encode()` crashes on Tiny Shakespeare (fixed in step 1).** It failed with
`RangeError: Maximum call stack size exceeded` at
`ids.push(...encode(pieces[i]))` (`src/BPETokenizer.ts:80`). Spreading an array
passes every element as a function argument, and this fails above about 125,000
tokens: it succeeds at 417,935 characters and fails at 422,214. The training
part of Tiny Shakespeare is 301,966 tokens. The same problem was already
avoided in `Trainer.joinDocuments()`.

**2. The Verdict cannot train on the large model (fixed in step 6).** Its validation part is 547
tokens, but a window on the large model needs 1,025.

**3. Reducing `contextLength` on a loaded model breaks saving (avoided by step 6).** The large model
is impractical with 1024-token windows, but lowering `contextLength` after
loading saves a value that no longer matches the position embedding, and
`loadModel()` then rejects the file. Step 6 fixes this and problem 2.

**4. Variable-length windows would be weighted wrongly (fixed in step 2).** `trainOnBatch()` scaled
each window's errors by `1 / window length`, then averages across windows. With
equal lengths this is the mean loss per token, which is what the trainer reports.
With unequal lengths each token in a short window counts for more than a token
in a long one, so the loss optimized is not the loss reported. Step 2 fixes this.

## A4. Measured sizes

Windows at each model's `contextLength` (training / validation):

| Dataset | Tokens (training / validation) | Small (128) | Medium (256) | Large (1024) |
|---|---|---|---|---|
| The Verdict | 4,763 / 547 | 37 / 4 | 18 / 2 | 4 / **0** |
| Tiny Shakespeare | 301,966 / 36,059 | 2,359 / 281 | 1,179 / 140 | 294 / 35 |

Alpaca has 52,002 examples, 28 of them with an empty output. With the prompt
template and end token:

| Longest example allowed | Examples kept | Tokens |
|---|---|---|
| 128 tokens | 36,477 (70%) | 3,041,212 |
| 256 tokens | 50,642 (97%) | 5,377,587 |
| No limit (longest is 1,511) | 52,002 | 5,833,242 |

All 52,002 examples tokenize in about 2 seconds, so the loader can simply
tokenize everything, with no need to read it lazily.

Estimated time for one pass over the training data, at the measured 110,000,
27,000 and 8,000 tokens an hour (desktop, Ryzen 7 3800X), not including
validation:

| Dataset | Small | Medium | Large |
|---|---|---|---|
| The Verdict | 3 minutes | 11 minutes | 36 minutes |
| Tiny Shakespeare | 2.7 hours | 11 hours | 38 hours |
| Alpaca, longest 128 tokens | 28 hours | 113 hours | 16 days |
| Alpaca, longest 256 tokens | n/a (context is 128) | 8 days | 28 days |

So the large model needs a way to train on part of a dataset, as does Alpaca on
any model.

---

# Part B: The design

## B1. Separate the data from the training loop

A new `Dataset` interface prepares windows, and `Trainer` trains on whatever
windows it is given:

```ts
// src/Dataset.ts
export interface Window {
  inputs: number[];
  targets: number[];
}

export interface DatasetWindows {
  training: Window[];
  validation: Window[];
}

export interface Dataset {
  /** Prepares the next windows to train on, using the current settings. */
  load(hyperparameters: Hyperparameters, tokenizer: BPETokenizer): DatasetWindows;
}
```

`src/Dataset.ts` also holds the shared helpers moved out of `Trainer`:
`createWindows(tokens, windowLength)` and `joinDocuments(documents, endOfTextId)`.

`Trainer.load(dataset)` replaces both of the original loaders:

```ts
trainer.load(new TextDataset("data/the-verdict.txt"));
trainer.load(new TextDataset("data/tinyshakespeare.txt", { maximumCharacters: 85000 }));
trainer.load(new TinyStoriesDataset("data/TinyStories_train.csv", "data/TinyStories_validation.csv"));
trainer.load(new AlpacaDataset("data/alpaca_data.json"));
trainer.train();
```

`train()`, `trainOnBatch()`, `shuffle()`, `createBatches()` and the validation
loss stay in `Trainer`, working from stored windows rather than tokens.

**Why not a `dataset` hyperparameter or a switch inside `Trainer`?** Each dataset
would add more branches to `Trainer`, and a fifth dataset would mean changing it
again. With this design, a new dataset is a new class and needs no `Trainer`
change. It also lets the tests give `Trainer` tiny datasets directly.

## B2. How each dataset behaves

| | `TextDataset` | `TinyStoriesDataset` | `AlpacaDataset` |
|---|---|---|---|
| **Used for** | The Verdict, Tiny Shakespeare | TinyStories | Alpaca |
| **Windows** | Fixed length, cut from continuous text | Fixed length, cut from stories joined by `<\|endoftext\|>` | One per example, varying in length |
| **Validation** | Last `1 - trainingRatio` of each file's characters | First `numberValidationExamples` of its own file | `numberValidationExamples` held out by a fixed shuffle |
| **Amount per `load()`** | The whole text, or `maximumCharacters` | The next `numberTrainingExamples` | The next `numberTrainingExamples` |
| **Repeated `load()` calls** | Same data | Continues to new stories | Continues to new examples |
| **Repeats within a run** | Through `numberEpochs` | One epoch is usual | One epoch is usual |

Each dataset object keeps its own position, replacing `Trainer.tinyStoriesLoaded`.
A new dataset object starts again from the beginning.

## B3. Alpaca's format

Each example becomes one text, using the original Alpaca templates, with the
input section left out when `input` is empty, followed by `<|endoftext|>`:

```text
Below is an instruction that describes a task, paired with an input that
provides further context. Write a response that appropriately completes the
request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}<|endoftext|>
```

The end token matters: it is how the model learns to stop answering.

- **Windows:** an example of `n` tokens gives one window of `n - 1` inputs, with
  the targets shifted one place.
- **Padding:** none is needed, because windows run through the model separately.
- **Long examples:** dropped, and the number dropped is reported. Truncating them
  would remove the end of the answer and the end token, teaching the model to
  stop mid-sentence.
- **Empty outputs:** the 28 examples are skipped.
- **Placeholder inputs:** about 600 examples have an input that only says there is
  none, such as `No input`, `<noinput>` or `N/A`. These use the template without an
  input section. This was found during step 7.
- **Validation split:** the examples are shuffled with a fixed seed held in
  `AlpacaDataset`, separate from `Hyperparameters.seed`. Validation examples come
  first and are removed from the training pool. The split is therefore the same
  in every run, and no validation example is ever trained on.
- **Prompts:** `AlpacaDataset.formatPrompt(instruction, input)` is public, so that
  chat can later use exactly the template the model was trained on.

## B4. New and renamed hyperparameters

| Name | Default | Purpose |
|---|---|---|
| `trainingWindowLength` | Unset: uses `contextLength` | The longest window to train on. Never above `contextLength`. |
| `numberTrainingExamples` | 1000 | Renamed from `numberTrainingStories`, as it now also counts Alpaca examples. |
| `numberValidationExamples` | 100 | Renamed from `numberValidationStories`. |

`trainingWindowLength` is the key addition. It does not change the model's shape,
so it is safe to change on a loaded model and is saved with the other training
settings. It fixes problems 2 and 3 in A3, and it also sets the longest Alpaca
example that is kept, so setting it to 128 on the large model trains on Alpaca's
shorter examples.

Saved models hold the old `numberTrainingStories` names. `fromJSON()` ignores
unknown values and uses the defaults, which is harmless because these are only
training amounts, so existing files such as `TinyStories-100.safetensors` still
load. `ModelManager.ARCHITECTURE` does not change.

---

# Part C: Decisions

All four recommendations below have been accepted.

1. **Add `trainingWindowLength`?** Recommended: yes. Otherwise The Verdict cannot
   train on the large model, and GPT-2 fine-tuning stays impractically slow.
2. **Drop or truncate long Alpaca examples?** Recommended: drop, for the reason in
   B3. With the small model this keeps 70% of the examples.
3. **Rename to `numberTrainingExamples` and `numberValidationExamples`?**
   Recommended: yes. Keeping the old names would be misleading for Alpaca, but it
   would avoid a good number of test edits.
4. **Keep the console on TinyStories for now?** Recommended: yes, as this plan
   covers `Trainer`. The console changes only as far as needed to keep working,
   and a dataset menu follows separately (see Part E).

---

# Part D: Steps

## Step 1: Fix the tokenizer crash (done)

- In `BPETokenizer.encode()`, replace `ids.push(...encode(pieces[i]))` with a loop,
  as in `Trainer.joinDocuments()`, with a comment explaining why.

**Check:** a new test encodes text of more than 150,000 tokens without an error,
and the ids decode back to the original text. `encode()` on the whole of Tiny
Shakespeare succeeds.

**Result:** the new test passes. The whole of Tiny Shakespeare encodes to 338,025
tokens in 0.14 seconds and decodes back to the identical text.

## Step 2: Weight the batch correctly for windows of different lengths (done)

- In `trainOnBatch()`, count the batch's tokens first, then scale every window's
  errors by `batch.length / tokens in the batch`, in place of
  `1 / window.targets.length`. The gradients are averaged over the windows, so the
  `batch.length` cancels, giving the gradient of the mean loss per token.
- Add a comment explaining this. For equal lengths the scale equals
  `1 / window length`, so nothing changes for existing datasets.

**Check:** the existing reference tests are unchanged, including the untrained
validation loss of `10.81001060108122`. A new test with two windows of different
lengths shows the batch's gradients match a numerical gradient of the reported
mean loss per token, using `checkGradients` from `GradientCheck.ts`.

**Result:** the new test passes; before the fix, 47 of its 48 values disagreed.
Seeded training at window lengths 4 and 7 gave identical losses and output-layer
weights before and after.

## Step 3: Add `Dataset` and `TextDataset`, and move `Trainer` to windows (done)

- New `src/Dataset.ts` as in B1, with `createWindows` and `joinDocuments` moved
  from `Trainer`.
- `Trainer` gets `load(dataset)` and stores the windows it returns. `train()`,
  `calculateValidationLoss()` and `checkValidationSet()` use them. The error
  messages are accurate for all datasets.
- Windows are now cut at `load()` rather than at `train()`, so a change to
  `contextLength` between the two no longer has any effect. No existing test relied
  on the old behaviour. The new behaviour is documented on `Trainer.load()` and has
  a test.
- New `src/TextDataset.ts`, moving the logic from `Trainer`, with the optional
  `maximumCharacters` taken from the start of each file before it is split.
- `Trainer.load()` takes only a `Dataset`.
- Move the text-file tests in `Trainer.test.ts` to the new API, and add
  `TextDataset.test.ts`.

**Check:** all tests pass, with the window counts from A4 for The Verdict
(37 / 4 at length 128). The reference validation loss is unchanged. Tiny
Shakespeare loads and gives 2,359 / 281 windows at length 128.

**Result:** all 288 tests pass, including the reference validation loss. Through
`Trainer.load()`, The Verdict gives 37 / 4 windows at length 128 and still 4 / 0 at
1024 (step 6 fixes this), and Tiny Shakespeare gives 2,359 / 281 at 128, or 181 / 21
with `maximumCharacters` 85,000. Seeded runs on text and on real TinyStories files
gave the same losses and output-layer weights as before the change. Tiny Shakespeare
has not yet been through a training run (step 8).

## Step 4: Add `TinyStoriesDataset` (done)

- New `src/TinyStoriesDataset.ts`, moving `loadTinyStories()` and `readStories()`
  from `Trainer`, with a public `storiesLoaded` count.
- Remove `Trainer.loadTinyStories()` and `tinyStoriesLoaded`.
- `GPTConsole` creates a dataset in `useModel()`, alongside the new trainer, so a
  replaced model still starts again from the first story. `writePlan()` reads
  `storiesLoaded`.

**Check:** all tests pass, including the console test that runs continue with
stories 1 to 10, 11 to 20 and 21 to 30. The console behaves as before.

**Result:** all tests pass, including that console test. The story tests moved to
`TinyStoriesDataset.test.ts`. Two seeded runs on the real TinyStories files, of 12
stories each, gave identical positions, losses and output-layer weights before and
after. The console itself was only run through its tests, not by hand.

## Step 5: Rename the story hyperparameters (done)

- Rename to `numberTrainingExamples` and `numberValidationExamples` throughout, as
  in B4, including `HyperparameterValues`, `toJSON()` and `fromJSON()`.

**Check:** all tests pass. `models/TinyStories-100.safetensors` still loads through
`ModelManager.loadModel()`.

**Result:** all 280 tests pass. The file holds the old names and loads with 13,675,520
parameters. Its example counts take the defaults of 1000 and 100, as B4 expects.

## Step 6: Add `trainingWindowLength` (done)

- Add it to `Hyperparameters`, following the existing style, with validation. The
  getter returns the smaller of the value set and `contextLength`, or
  `contextLength` when unset. Like `seed`, it is left out of `toJSON()` when unset.
- The datasets use it in place of `contextLength`.

**Check:**
- The Verdict trains on a GPT-2-shaped model with `trainingWindowLength` 256, where
  it previously failed for having no validation windows.
- A model with `trainingWindowLength` 128 and `contextLength` 1024 saves and
  reloads through `loadModel()`, keeping both values, which is the save problem
  from A3.
- Leaving it unset changes nothing: the reference validation loss is unchanged.

**Result:** all three hold.
- The Verdict gives 18 / 2 windows at 256 on a full-size GPT-2-shaped model with
  random weights. A full epoch would take about 35 minutes, so only one batch and the
  validation were run, taking 328 seconds with finite losses.
- The save and reload was tested on a tiny model with the same 128 and 1024 settings,
  not on a full-size one.
- Seeded runs with it unset gave the same losses and output-layer weights as before.

`GPTConsole.storiesForHours()` also uses it now, for the fewest stories that make two
batches. A setting above `contextLength` is kept as set in the saved file, but the
getter limits it to `contextLength`.


## Step 7: Add `AlpacaDataset` (done)

- New `src/AlpacaDataset.ts` as in B3. Reading the JSON is a plain
  `JSON.parse(fs.readFileSync(...))`, as the file is only 23 MB.
- If fewer examples remain than requested, use what remains and warn, as
  `readStories()` does.
- Print the number of examples kept, dropped as too long and skipped as empty when
  `DEBUG.TRAINER` is on.

**Check:** new `AlpacaDataset.test.ts`, on a small temporary JSON file:
- both templates are formatted exactly;
- each window's targets are its inputs shifted by one, and the last target is the
  end token;
- examples that are too long or have empty outputs are left out and counted;
- no validation example appears in the training windows;
- the split is the same for any `Hyperparameters.seed`;
- a second `load()` gives new training examples and the same validation examples;
- on the real file with `trainingWindowLength` 128, about 36,500 examples are kept.
  The exact figure differs slightly from A4, because an example of `n` tokens needs
  a window of `n - 1` and the 28 empty outputs are removed.

**Result:** all 310 tests pass, including all of the above except the real-file count,
which was run separately. On the real file:

| Longest window | Kept | Dropped as too long | Skipped as empty |
|---|---|---|---|
| 128 | 36,826 | 15,148 | 28 |
| 256 | 50,637 | 1,337 | 28 |
| 1024 | 51,972 | 2 | 28 |

The file is read and tokenized in 1.8 seconds. These differ slightly from A4 (36,477
and 50,642), which counted whole examples rather than windows, included the empty
outputs and used the longer template for placeholder inputs. A tiny model trained on 20 real examples (windows of
57 to 127 tokens) with finite losses. That was too short to show learning, which is
step 8's job.

A file position is kept in the file's shuffled order, so changing `trainingWindowLength`
between loads does not repeat examples already trained on. Examples skipped earlier as
too long are not revisited if it is then raised. `countExamples()` was added to report
the kept, dropped and skipped counts.

## Step 8: A real run on each dataset

A short run on each with the small model, through a temporary script rather than
the console, checking only that it learns: training and validation loss fall, and
samples begin to resemble the data. This takes about 40 minutes in total.

| Dataset | Amount | Approximate time |
|---|---|---|
| The Verdict | 1 epoch | 3 minutes |
| Tiny Shakespeare | `maximumCharacters` 85,000 (about 25,000 tokens), 1 epoch | 13 minutes |
| TinyStories | 100 stories | 14 minutes, including validation |
| Alpaca | 200 examples | 10 minutes |

**Check:** no errors, and the loss falls on all four. Record the results here.

**Result:** no errors on any of the four, using a fresh small model with seed 42, 38.7
minutes in total. An untrained model's loss is about 10.8.

| Dataset | Training windows | Validation loss before | Validation loss after | Time |
|---|---|---|---|---|
| The Verdict | 37 | 10.840 | 8.235 | 2.5 minutes |
| Tiny Shakespeare | 181 | 10.842 | 5.338 | 12.1 minutes |
| TinyStories | 147 | 10.849 | 5.877 | 14.2 minutes |
| Alpaca | 200 | 10.840 | 4.144 | 9.9 minutes |

- **Training loss:** a bug in the measuring script, which was not in the project's code,
  lost the batch losses. After the fix, a repeat run on The Verdict gave the same
  validation loss and fell from 10.692 over the first 2 batches to 8.667 over the last 2.
  The other three were not repeated, so their training loss was not recorded.
- **Comparing the losses:** these are not comparable between datasets. Alpaca's is
  flattered because much of each window is the fixed template.
- **Samples:** only the beginnings of learning, as expected from runs this short. Greedy
  sampling mostly repeats the commonest token (`,` or a line break), and top-k gives
  common words without sentences. Tiny Shakespeare has started on speaker names, and
  Alpaca repeats `### Response:` and has produced `<|endoftext|>`, but gives no answer.

## Step 9: Documentation (done)

- Update the comments on `Trainer`, `Hyperparameters` and `GPTConsole`.
- Add a note to `TRAINING_PLAN.md` that the loaders have moved.
- Every new class and method needs API-style doc comments, per `CLAUDE.md`.

---

# Part E: Not part of this plan

These are needed for the full demonstrations, but do not belong to `Trainer`:

- **Choosing the dataset in the console.** Done since this plan: the console has
  Pretrain and Fine tune options that offer the datasets, and `TrainingPlanner` has
  replaced `GPTConsole.storiesForHours()`.
- **Stopping at the end token.** `Generator` always produces `outputLength` tokens,
  so an Alpaca-trained model would run on past its answer.
- **A prompt mode in chat,** wrapping what the user types in the Alpaca template
  with `AlpacaDataset.formatPrompt()` and showing only the response.
- **Checkpoints during training.** `train()` saves nothing until it returns, which
  is a serious risk on a 50-hour run. AdamW's running averages are also not saved,
  so a resumed run restarts the optimizer.
- **Fine-tuning settings.** Done since this plan: `TrainingPlanner` uses a learning
  rate of 0.00005 and a batch size of 8 for fine tuning.
