"""Milestone 5: train on Tiny Shakespeare on a Kaggle T4.

Data (about 1 MB):
  https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

Write the loop yourself. It must do all of the following:

  1. Read the text, build a CharTokenizer, encode to one long tensor.
  2. Split 90% train / 10% validation. Split by position, not at random.
     (Question: why would a random split of windows leak?)
  3. Build GPT(GPTConfig(...)) and move it to the GPU.
  4. AdamW, learning rate 3e-4 as a starting point.
  5. Each step: get_batch -> forward -> zero_grad -> backward -> step.
  6. Every 500 steps, estimate train and validation loss by averaging over
     about 50 batches under torch.no_grad() with model.eval(). Print both.
  7. Save a checkpoint (model state, optimizer state, step, config) and be
     able to resume from it. Kaggle sessions die; you will need this.
  8. At the end, generate 500 characters and print them.

Target: validation loss below 1.6 with the default GPTConfig and block_size=128
in under 20 minutes on a T4. If the first loss you print is not close to
ln(vocab_size), which is about 4.17 for 65 characters, your model or loss is
wrong. Stop and find out why before you train further.
"""

raise NotImplementedError("Write the training loop described above.")
