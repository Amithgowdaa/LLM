# Study notes: building a GPT from scratch

These notes cover the theory behind every milestone in this repo, plus the
compute plan. They explain the ideas, the tensor shapes and the common bugs.
They do not contain the code. You write that.

How to use them:

- Read the section for a milestone **before** you open its stub file.
- Each section ends with "Check yourself" questions. Answer them on paper first.
  Short answers are in the last section. If you read the answers before trying,
  you have wasted the exercise. Nobody is checking but you.

---

## Contents

0. [The plan and the compute you have](#0-the-plan-and-the-compute-you-have)
1. [What a language model is](#1-what-a-language-model-is)
2. [Milestone 1: tokenizer and batches](#2-milestone-1-tokenizer-and-batches)
3. [Milestone 2: causal self-attention](#3-milestone-2-causal-self-attention)
4. [Milestone 3: MLP, residuals and LayerNorm](#4-milestone-3-mlp-residuals-and-layernorm)
5. [Milestone 4: the full GPT](#5-milestone-4-the-full-gpt)
6. [Milestone 5: training](#6-milestone-5-training)
7. [Memory and speed arithmetic](#7-memory-and-speed-arithmetic)
8. [After milestone 5](#8-after-milestone-5)
9. [Overlap with GATE](#9-overlap-with-gate)
10. [Reading list](#10-reading-list)
11. [Glossary](#11-glossary)
12. [Answers to "Check yourself"](#12-answers-to-check-yourself)

---

## 0. The plan and the compute you have

### Project order

| Order | Project | Why it matters | Compute |
|---|---|---|---|
| 1 | **minigpt** (this repo) | You understand what every later tool hides | Laptop CPU; Kaggle T4 for ~20 minutes |
| 2 | **Domain fine-tune with evaluation** (QLoRA on a 1–8B model) | The project that wins freelance work | Kaggle T4 |
| 3 | **Large-scale data pipeline** (Ray or Spark, 50–500 GB) | Large data and parallelism | Azure for Students CPU VMs + Blob Storage |
| 4 | **Distributed training** (PyTorch DDP) | Multi-GPU parallelism | Kaggle T4 ×2 |

GATE comes first. At about 3 hours a week, minigpt alone takes 2–3 months.
That is fine. Slow and understood beats fast and copied.

### Compute

- **Kaggle** (free): T4 ×2 or P100, roughly 30 GPU hours a week, 12-hour
  sessions. This covers every GPU need in the plan, including DDP, because
  T4 ×2 is two GPUs on one machine.
- **Colab free**: T4 when available; disconnects often. A backup only.
- **SageMaker Studio Lab**: free T4 with short sessions; needs approval. A backup.
- **Azure for Students**: about $100 credit for 12 months. GPU quota is
  usually 0. Use it for CPU-heavy work in project 3. Set a $10 budget alert
  on day one and stop every VM when you finish.

Quotas change. Check each provider's page before relying on these numbers.

---

## 1. What a language model is

A language model assigns a probability to a sequence of tokens. A GPT does it
by predicting one token at a time, using the **chain rule of probability**:

    P(x_1, x_2, ..., x_n) = P(x_1) · P(x_2 | x_1) · P(x_3 | x_1, x_2) · ... · P(x_n | x_1..x_{n-1})

This is exact, not an approximation. Any joint distribution can be factored
this way. The model's job is to learn each conditional `P(x_t | x_<t)`.

**Training objective.** Maximise the probability of the real text. Taking logs
turns the product into a sum, and negating it gives the loss:

    loss = -(1/n) · Σ_t log P(x_t | x_<t)

This is the **cross-entropy** between the true next token (a one-hot vector)
and the model's predicted distribution. Minimising it is maximum likelihood
estimation, the same MLE you study in GATE probability.

**Why "autoregressive".** At generation time the model's own output becomes
its next input. It samples a token, appends it, and predicts again.

**Why "causal".** When predicting token `t`, the model may only look at
tokens `< t`. During training we feed whole sequences at once for speed, so
the architecture itself must stop each position from seeing the future. That
is the causal mask in milestone 2.

**Reading the loss.** The loss is in nats (natural log).

- A model that guesses uniformly over `V` tokens has loss `ln V`. For 65
  characters that is `ln 65 ≈ 4.17`. This is your first sanity check.
- `exp(loss)` is the **perplexity**: roughly, "the model is as confused as if
  it were choosing uniformly among this many tokens". Loss 1.5 means
  perplexity ≈ 4.5.

### Check yourself
1.1 Why is a loss of 0 impossible on real text, even for a perfect model?
1.2 If your untrained model prints loss 11.3 on a 65-character vocabulary, what does that tell you?

---

## 2. Milestone 1: tokenizer and batches

### Tokenization

A neural network needs numbers, not text. A **tokenizer** maps text to a list
of integer ids and back.

- **Character-level** (what you build): every distinct character is a token.
  Tiny Shakespeare has 65 of them. Simple, no unknown tokens, but sequences
  are long and the model must learn spelling from scratch.
- **Subword (BPE)**, used by real LLMs: frequent character sequences become
  single tokens (`" the"`, `"ing"`). Vocabularies of 32k–200k. Shorter
  sequences, so the same context window holds more text. You build this after
  milestone 5.

Your vocabulary must be built in **sorted** order. If you use the order of
`set()`, ids can change between runs, and a saved checkpoint becomes useless
with a freshly built tokenizer. Determinism is a correctness property here,
not a style preference.

### From one long text to training examples

Encode the whole text into one 1-D tensor `data` of length `N`. A training
example is a window of `T` consecutive tokens (`T = block_size`).

    data:  [ 5, 12, 40,  7, 33, 21, 18, ... ]
    x:     [ 5, 12, 40,  7 ]           ← window starting at i
    y:     [12, 40,  7, 33 ]           ← same window shifted by one

`y[t]` is the token that follows `x[t]`. One window of length `T` therefore
gives **T training examples at once**: predict `12` from `[5]`, predict `40`
from `[5, 12]`, and so on up to predicting `33` from all four. The causal mask
is what makes this legal.

A batch stacks `B` such windows starting at random positions, giving `x` and
`y` of shape `(B, T)`. The dtype must be `torch.long`, because
`nn.Embedding` and `F.cross_entropy` require integer indices.

### Random start indices

`torch.randint(low, high, size)` returns values in `[low, high)`: **high is
exclusive**. This is the single most common off-by-one in this milestone.
Work out the largest valid start from the fact that `y` reads one position
further than `x`.

### The `generator` argument

Passing a `torch.Generator` makes sampling reproducible without touching the
global random seed. You will need this when you compare two runs and want them
to see identical batches.

### Check yourself
2.1 For `data` of length `N` and block size `T`, what is the largest valid start index? Why is it not `N − T`?
2.2 In `train.py` you split train/validation by position (first 90% / last 10%). Why would sampling random windows and assigning them randomly to train or validation leak?
2.3 With `N = 1,000,000` and `T = 128`, roughly how many distinct windows exist? Does one "epoch" mean anything when sampling randomly?

---

## 3. Milestone 2: causal self-attention

This is the core of the transformer. Spend the most time here.

### The idea

Each position needs information from earlier positions, but which ones matter
depends on the content. In "The cat sat because **it** was tired", the token
"it" should attend to "cat". Attention lets every position compute a weighted
average of earlier positions, with weights computed from the data.

Each position produces three vectors from its embedding `x` by three learned
linear maps:

- **Query** `q`: "what am I looking for?"
- **Key** `k`: "what do I contain?"
- **Value** `v`: "what do I hand over if you pick me?"

The score between position `i` (query) and position `j` (key) is the dot
product `q_i · k_j`. Large score = good match.

### Scaled dot-product attention

For a single head with dimension `d_k`:

    scores  = Q Kᵀ / √d_k          shape (T, T)
    scores  = mask(scores)          future positions → −∞
    weights = softmax(scores)       each row sums to 1
    out     = weights · V           shape (T, d_k)

Row `i` of `weights` is a probability distribution over positions `0..i`.
Row `i` of `out` is the weighted average of the value vectors at those positions.

### The causal mask

A lower-triangular matrix of ones: `mask[i, j] = 1` if `j ≤ i`, else 0. Where
the mask is 0, set the score to `−∞` **before** the softmax. Since
`exp(−∞) = 0`, those positions get exactly zero weight, and the remaining
weights still sum to 1.

Store the mask with `self.register_buffer(...)`, not as a plain attribute and
not as an `nn.Parameter`. A buffer moves to the GPU with `.to(device)` and is
saved in `state_dict`, but it is not trained. Build it at size
`block_size × block_size` once, then slice `[:T, :T]` in `forward`, because
the input can be shorter than `block_size`, especially during generation.

### Multi-head attention

One head can only learn one kind of "matching". With `h` heads, split the
embedding dimension `C` into `h` pieces of size `C / h`. Each head attends
independently, so one head might track the previous character, another the
start of the current word. Concatenate the head outputs back to size `C`,
then apply one more linear layer (the output projection).

The cost is the same as one big head: the total dimension is still `C`.

### The shapes, step by step

Track these on paper. Nearly every attention bug is a shape bug.

| Step | Shape |
|---|---|
| input `x` | `(B, T, C)` |
| project to q, k, v (one `Linear(C, 3C)` or three `Linear(C, C)`) | 3 × `(B, T, C)` |
| split heads: view as `(B, T, h, C/h)` | `(B, T, h, d)` with `d = C/h` |
| move heads forward: transpose dims 1 and 2 | `(B, h, T, d)` |
| scores = q @ kᵀ (transpose the last two dims of k) | `(B, h, T, T)` |
| mask, then softmax over the **last** dim | `(B, h, T, T)` |
| weights @ v | `(B, h, T, d)` |
| transpose back, merge heads | `(B, T, C)` |
| output projection | `(B, T, C)` |

`@` (matmul) on 4-D tensors multiplies the last two dimensions and treats the
leading ones as batch dimensions.

### Two PyTorch traps

- **`view` after `transpose` fails** with "view size is not compatible with
  input tensor's size and stride". `transpose` does not move data in memory;
  it only changes strides, so the tensor is not contiguous. Use `.reshape`
  or call `.contiguous()` before `.view`.
- **Softmax over the wrong dimension** gives no error, just a model that never
  learns. Each query's weights must sum to 1, so softmax goes over the key
  dimension, the last one.

### Cost

The scores tensor is `T × T` per head. Time and memory grow with `T²`. That
is why context length is expensive, and why later work (FlashAttention, KV
caching) focuses on this step.

### Check yourself
3.1 If the entries of `q` and `k` are independent with mean 0 and variance 1, what is the variance of `q · k` for dimension `d_k`? Use this to explain why we divide by `√d_k`.
3.2 Why must the mask be applied before the softmax and not after?
3.3 What is the shape of the scores tensor for `B = 2, T = 8, h = 4`?
3.4 Self-attention without position information is **permutation-equivariant**. What does that mean, and why is it a problem for language?

---

## 4. Milestone 3: MLP, residuals and LayerNorm

Attention moves information **between** positions. The MLP processes
information **within** each position, independently. A transformer block
alternates the two.

### The MLP (feed-forward network)

    Linear(C → 4C)  →  GELU  →  Linear(4C → C)  →  Dropout

- The 4× expansion is a convention from the original transformer paper. Most
  of a transformer's parameters live here.
- **GELU** is `x · Φ(x)`, where `Φ` is the standard normal CDF. It is a smooth
  version of ReLU and is what GPT-2 uses. You may call `F.gelu` or
  `nn.GELU`.
- `nn.Linear` on a `(B, T, C)` tensor applies the same weights at every
  position. No reshaping is needed.

### Residual connections

    x = x + sublayer(x)

Instead of each block computing a new representation, it computes an
**update** that is added to a running "residual stream".

- Gradients flow straight through the `+` back to early layers, so deep
  networks train. Without it, the gradient must pass through every sublayer's
  Jacobian, and it shrinks or explodes with depth.
- A block can do nothing (output `x`) by making its sublayer output 0, which
  is easy to learn. The test `test_block_has_a_gradient_path_through_the_residual`
  checks this: with all weights zeroed, the block must return its input.

### LayerNorm

For each position separately, normalise the `C` features to mean 0 and
variance 1, then apply a learned scale and shift:

    y = (x − mean) / sqrt(var + ε) · γ + β

- It normalises over the **feature** dimension, per token. BatchNorm
  normalises over the batch, which behaves badly with variable sequence
  lengths and small batches. That is why transformers use LayerNorm.
- It keeps activations in a stable range as depth grows.

### Pre-norm vs post-norm

- **Post-norm** (original 2017 transformer): `x = LN(x + sublayer(x))`
- **Pre-norm** (GPT-2 onward, what you build): `x = x + sublayer(LN(x))`

Pre-norm leaves the residual stream untouched by normalisation, so gradients
pass through cleanly. It trains more stably without careful learning-rate
warmup. Because the stream itself is never normalised, the model needs **one
final LayerNorm** after the last block (milestone 4).

### Dropout

Randomly zeroes a fraction `p` of activations during training and scales the
rest by `1/(1−p)`. At evaluation it does nothing. This is why `model.eval()`
matters. The default config uses `dropout = 0.0`; for a small dataset like
Shakespeare you may try 0.1–0.2 once the model overfits (validation loss
rising while training loss falls).

### Check yourself
4.1 What breaks in a deep network if you remove the two `x +` terms?
4.2 Why does a pre-norm model need a final LayerNorm, but a post-norm model does not?
4.3 Roughly what fraction of a block's weights are in the MLP versus the attention layer? (Count the `Linear` layers.)

---

## 5. Milestone 4: the full GPT

### Architecture

    idx (B, T) token ids
      → token embedding   (B, T, C)    nn.Embedding(vocab_size, C)
      + position embedding (T, C)      nn.Embedding(block_size, C), indexed by 0..T-1
      → n_layer × Block
      → final LayerNorm
      → lm_head: Linear(C → vocab_size)
      → logits (B, T, vocab_size)

### Position embeddings

From 3.4: attention alone cannot tell position 2 from position 7. Adding a
learned vector per position fixes this. Create the position indices with
`torch.arange(T, device=idx.device)`. If you forget `device=`, you get a
"tensors on different devices" error on the GPU but not on the CPU, so the
bug hides until milestone 5.

The `(T, C)` position embedding adds to the `(B, T, C)` token embedding by
**broadcasting**.

Modern LLMs use **RoPE** (rotary position embedding) instead of learned
positions. It rotates q and k by a position-dependent angle so that the dot
product depends on relative distance. Learn it after milestone 5.

### Logits and loss

`F.cross_entropy` expects logits of shape `(N, V)` and targets of shape `(N,)`.
Your logits are `(B, T, V)`, so flatten to `(B·T, V)` and targets to `(B·T,)`.
Pass **raw logits**, not softmax output: `cross_entropy` applies log-softmax
internally, in a numerically stable way.

### Initialisation

The test `test_initial_loss_is_near_uniform` requires the untrained loss to
be close to `ln(vocab_size)`. That only happens if the initial logits are
near zero, so the softmax is near uniform. PyTorch's default initialisation
can give larger logits. GPT-2 initialises `Linear` and `Embedding` weights
from a normal distribution with standard deviation 0.02 and biases at 0.
Loop over `self.modules()` and apply that.

A high initial loss is not harmless: the model wastes its first hundreds of
steps just shrinking the logits.

### Weight tying (optional)

The token embedding is `(V, C)` and the lm_head weight is also `(V, C)`. GPT-2
shares one matrix for both. It saves parameters and often helps a little.
Try it after the tests pass and compare validation loss.

### Generation

Repeat `max_new_tokens` times:

1. Crop the context to the last `block_size` tokens. The position embedding
   has no row for position `block_size`, so feeding a longer sequence crashes.
2. Run the model and keep only the logits at the **last** position: `(B, V)`.
3. Divide by `temperature`.
   - `< 1`: sharper distribution, safer and more repetitive text.
   - `> 1`: flatter distribution, more random text.
   - As it approaches 0, sampling becomes greedy (always the argmax).
4. Optional **top-k**: keep the `k` largest logits, set the rest to `−∞`.
   `top_k = 1` is greedy decoding.
5. Softmax, then sample one token with `torch.multinomial`.
6. Append it with `torch.cat` along dimension 1.

`@torch.no_grad()` is already on `generate`: no gradients are needed, which
saves memory and time.

This loop recomputes the whole context every step: O(T²) work per token. The
KV cache (section 8) fixes that.

### Parameter count

Per block, ignoring biases and LayerNorm: attention has 4 matrices of `C × C`
(q, k, v, output) and the MLP has 2 of `C × 4C`. That is `4C² + 8C² = 12C²`.

    total ≈ 12 · n_layer · C²  +  V·C (token emb)  +  block_size·C (pos emb)  +  C·V (head)

For the default config (`C = 128`, 4 layers, `V = 65`): `12 · 4 · 16,384 ≈ 786k`
in the blocks, plus about 25k elsewhere, so **about 0.8M parameters**. Check
your model with `sum(p.numel() for p in model.parameters())`. If the number
is far off, something is wired wrong.

GPT-2 small: `C = 768`, 12 layers, `V = 50,257` → about 124M. Check that the
formula gets you close.

### Check yourself
5.1 Why does generation crop to `block_size` instead of feeding the whole history?
5.2 Why pass logits rather than probabilities to `F.cross_entropy`?
5.3 With `temperature = 1` and no top-k, how often should the model choose a token it assigns probability 0.01?

---

## 6. Milestone 5: training

### The loop

    for step in range(max_steps):
        x, y = get_batch(train_data, ...)
        logits, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

That is the whole algorithm. Everything else is measurement and safety.

### AdamW

Adam keeps, per parameter, a running mean of gradients (momentum) and a
running mean of squared gradients (to scale each parameter's step). AdamW
applies weight decay directly to the weights instead of through the gradient,
which is the correct way to combine weight decay with Adam.

- Learning rate 3e-4 is a safe start for a model this size. If the loss is
  noisy or explodes, lower it; if it falls very slowly, try 1e-3.
- Later: a **warmup** (linearly increase the learning rate for the first few
  hundred steps) then **cosine decay** usually gives a better final loss.

### Evaluating the loss properly

The loss on one batch is noisy. To compare train and validation:

1. `model.eval()` (turns off dropout).
2. Under `torch.no_grad()`, average the loss over about 50 batches from each split.
3. `model.train()` again before continuing.

Forgetting step 3 is a classic bug: dropout stays off for the rest of training.

### Reading the curves

| Train loss | Val loss | Diagnosis | Action |
|---|---|---|---|
| Falling | Falling, close to train | Healthy | Keep going |
| Falling | Flat or rising | Overfitting | Dropout, smaller model, stop earlier |
| Flat, high | Flat, high | Underfitting or bug | Bigger model, higher LR, check the code |
| Starts ≫ ln V | — | Initialisation or loss bug | Fix before training |
| Becomes NaN | — | LR too high, or fp16 overflow | Lower LR, check the GradScaler |

### Sanity checks, in this order

1. **Initial loss ≈ ln V** (≈ 4.17).
2. **Overfit one batch**: train on a single fixed batch; loss should go near 0.
   The test `test_can_overfit_one_batch` does this. If a model cannot memorise
   64 tokens, it has a bug, not a capacity problem.
3. **Generated text improves** from random characters, to word-like shapes,
   to Shakespeare-like lines with names and line breaks.

### Checkpoints

Kaggle sessions end without warning. Save a dictionary with:

- `model.state_dict()`
- `optimizer.state_dict()` (Adam's moments; without them, resuming behaves
  like a restart with a cold optimizer)
- the step number
- the config
- ideally the RNG state, for exact resumption

Save with `torch.save`, load with `torch.load(..., map_location=device)`.
On Kaggle, write to `/kaggle/working/` so the file survives as notebook output.

### Using the T4

- `device = "cuda" if torch.cuda.is_available() else "cpu"`; move the model
  and every batch to it.
- The T4 has 16 GB of memory and fast **fp16** tensor cores. It does **not**
  have fast bf16 support. For mixed precision on a T4 use
  `torch.autocast("cuda", dtype=torch.float16)` together with
  `torch.amp.GradScaler`, which scales the loss to stop small fp16 gradients
  underflowing to zero. Get the plain fp32 version working first.
- If the GPU sits idle (check `nvidia-smi`), the bottleneck is the Python
  data loading, not the model. For this dataset, keep the whole tensor on the
  GPU and slice from it.

### About the target

`train.py` says "validation loss below 1.6 in under 20 minutes". I set that
target without training this exact configuration, so treat it as an estimate.
If you plateau around 1.7–1.8 with the default config, that is not
necessarily a bug. Try `block_size = 128`, `n_embd = 256`, `n_layer = 6`,
`dropout = 0.1`, and record what each change does. Changing one thing at a
time and writing down the result is the actual research skill here.

### Check yourself
6.1 Why must `zero_grad` be called every step? What happens if you skip it?
6.2 Your train loss is 1.2 and validation loss is 1.9 and rising. What happened and what do you change?
6.3 Why does resuming without the optimizer state hurt, even though the model weights are restored?

---

## 7. Memory and speed arithmetic

You will need this for the fine-tuning and distributed projects. Learn it now
while the numbers are small.

### Training memory per parameter (fp32, Adam)

| Item | Bytes per parameter |
|---|---|
| Weights | 4 |
| Gradients | 4 |
| Adam first moment | 4 |
| Adam second moment | 4 |
| **Total** | **16** |

Plus **activations**, which grow with batch size × sequence length × layers.
The `(B, h, T, T)` attention scores are often the largest single item.

Examples:

- minigpt (0.8M params): 0.8M × 16 B ≈ 13 MB. Trivial.
- A 1B model: about 16 GB before activations. That is already more than a T4
  holds. This is why full fine-tuning of even small LLMs on a T4 is not
  possible, and why the fine-tuning project uses **QLoRA**: the base weights
  are frozen and stored in 4-bit, and only small adapter matrices are trained.

### Inference memory

Weights only: params × bytes per param. A 7B model is 14 GB in fp16, about
3.5 GB in 4-bit. Plus the KV cache (section 8).

### Compute

A forward and backward pass costs about `6 × params × tokens` floating-point
operations. A T4 delivers roughly 65 TFLOPS of fp16 tensor-core compute at
peak; real training reaches a fraction of that. Use this to estimate how long
a run will take before you start it.

---

## 8. After milestone 5

### Milestone 6: byte-pair encoding (BPE)

Start from bytes (256 base tokens). Repeatedly find the most frequent adjacent
pair of tokens in the training text and merge it into a new token. Stop at the
target vocabulary size. Encoding applies the merges in the order they were
learned. Things to discover yourself: why start from bytes rather than
characters, and why does GPT-2 split text into words before merging?

### Milestone 7: KV cache

During generation, the keys and values of past tokens never change. Store them,
and on each new step compute q, k and v only for the **new** token, then attend
over the cached k and v. Work per new token drops from O(T²) to O(T). Cost:
memory of `2 × n_layer × T × C` values per sequence. Measure tokens per second
before and after; that number goes in your README.

### Milestone 8: mixed precision and gradient accumulation

- **Mixed precision**: see section 6. Measure speed and memory against fp32.
- **Gradient accumulation**: run `k` small batches, call `backward` on each
  (dividing the loss by `k`), and only then call `optimizer.step()`. This
  simulates a batch `k` times larger without the memory. It is the same idea
  that distributed data parallelism uses across GPUs, which is why it is the
  bridge to the DDP project.

---

## 9. Overlap with GATE

This project is not time stolen from GATE. Much of it is GATE material in use:

| This project | GATE topic |
|---|---|
| Matrix products, shapes, `Q Kᵀ` | Linear algebra: matrix multiplication, transpose, rank |
| Softmax, cross-entropy, MLE, sampling | Probability: distributions, expectation, likelihood |
| Variance of a dot product (3.1) | Probability: variance of sums of independent variables |
| Gradients, chain rule, backprop | Calculus: partial derivatives, chain rule |
| O(T²) attention, KV cache | Algorithms: time and space complexity |
| Memory per parameter, GPU vs CPU | Computer organisation: memory hierarchy, number representation |
| fp16 vs fp32 overflow and underflow | Computer organisation: floating-point representation |
| Data loading bottlenecks | Operating systems: processes, I/O |
| BPE merges, tokenizer | Data structures: hash maps, priority queues |

When a GATE topic shows up here, write a one-line note linking the two. It
helps both.

---

## 10. Reading list

Read in this order. Do not read source code of other implementations until
your own version passes the tests.

1. **Vaswani et al., "Attention Is All You Need" (2017).** Sections 3.1–3.3
   only, for now. Note it uses post-norm and an encoder-decoder; you build a
   decoder-only, pre-norm model.
2. **Radford et al., "Language Models are Unsupervised Multitask Learners"
   (GPT-2, 2019).** Section 2.3 lists the changes from GPT-1, including
   pre-norm and the 0.02 initialisation.
3. **Jay Alammar, "The Illustrated Transformer"** (blog post). Good pictures
   for section 3. Read it if the shapes table does not click.
4. **After milestone 4 passes:** Andrej Karpathy's lecture "Let's build GPT:
   from scratch, in code, spelled out". Compare it with what you wrote. Where
   his choices differ from yours, work out why.
5. **Later:** Su et al., "RoFormer" (RoPE, 2021); Hu et al., "LoRA" (2021);
   Dettmers et al., "QLoRA" (2023). These lead into the fine-tuning project.

---

## 11. Glossary

- **Token**: the unit the model reads; here, one character.
- **Vocabulary (V)**: the set of all tokens.
- **Block size / context length (T)**: the maximum number of tokens the model sees at once.
- **Embedding dimension (C, n_embd, d_model)**: the length of the vector representing each token.
- **Logits**: unnormalised scores before softmax.
- **Head**: one independent attention computation of dimension `C / h`.
- **Residual stream**: the running `(B, T, C)` tensor that every block adds to.
- **Causal mask**: lower-triangular mask that hides future positions.
- **Perplexity**: `exp(loss)`.
- **Buffer**: a tensor saved with the model and moved with `.to()`, but not trained.
- **Checkpoint**: a saved snapshot of model, optimizer and step, for resuming.
- **Mixed precision**: computing in fp16 while keeping master weights in fp32.
- **KV cache**: stored keys and values of past tokens, reused during generation.
- **LoRA / QLoRA**: fine-tuning by training small low-rank adapter matrices, with the base model frozen (and 4-bit in QLoRA).

---

## 12. Answers to "Check yourself"

Stop. Have you written your own answers on paper? If not, go back.

**1.1** Real text is not deterministic: many next characters are plausible. The
lowest achievable loss is the entropy of the text, which is above 0.

**1.2** Far above `ln 65 ≈ 4.17`, so the initial logits are too large. Your
initialisation is wrong or you are computing the loss on the wrong tensors.

**2.1** `N − T − 1`. The target window reads up to index `start + T`, which must
be at most `N − 1`. With `torch.randint`, whose upper bound is exclusive, the
`high` argument is therefore `N − T`. A start of `N − T` itself would make `y`
one token short.

**2.2** Neighbouring windows overlap almost completely. A validation window
would share most of its text with training windows, so validation loss would
measure memorisation, not generalisation.

**2.3** About `N − T ≈ 999,872` start positions. With random sampling, "epoch"
just means "about N tokens seen"; there is no guarantee every window was used.

**3.1** The variance is `d_k` (a sum of `d_k` independent products, each of
variance 1). With `d_k = 64`, scores have standard deviation 8, the softmax
becomes nearly one-hot, and its gradients become tiny. Dividing by `√d_k`
brings the variance back to 1.

**3.2** If you zero future weights after the softmax, the future scores were
already in the normalising denominator, so information leaks, and the
remaining weights no longer sum to 1. Setting `−∞` before the softmax gives
them exactly zero weight and a valid distribution over the past.

**3.3** `(2, 4, 8, 8)`: batch, heads, query positions, key positions.

**3.4** Shuffling the input positions just shuffles the outputs the same way.
The model cannot tell "dog bites man" from "man bites dog". Position
embeddings break the symmetry.

**4.1** Gradients must pass through every sublayer, so they vanish or explode
with depth, and each block must relearn the identity to preserve information.
Deep stacks train badly or not at all.

**4.2** In pre-norm, the residual stream is never normalised, so its scale grows
across blocks; the final LayerNorm fixes it before the output head. In
post-norm, every block already ends in a LayerNorm.

**4.3** About two thirds MLP (`8C²`) and one third attention (`4C²`).

**5.1** The position embedding only has `block_size` rows, and the model was
never trained on longer contexts.

**5.2** `cross_entropy` applies log-softmax itself, in a numerically stable way.
Feeding probabilities applies softmax twice and also loses precision.

**5.3** About 1 time in 100. Over 500 generated characters, low-probability
choices happen several times, which is why top-k or a lower temperature makes
text look cleaner.

**6.1** PyTorch **adds** new gradients to `.grad`. Without `zero_grad`, each
step uses the sum of all past gradients, and training diverges.

**6.2** Overfitting. Add dropout, reduce model size, or stop at the step with
the lowest validation loss (keep that checkpoint).

**6.3** Adam's moment estimates restart at zero, so the first steps after
resuming have badly scaled updates. The loss can jump, and the run is no
longer equivalent to an uninterrupted one.
