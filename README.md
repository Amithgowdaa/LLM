# minigpt

A GPT written from scratch, one milestone at a time. The files contain stubs
and questions; the tests tell you when a milestone is done.

Theory for every milestone is in [NOTES.md](NOTES.md). Read the section before you start the milestone.

## Rules

1. Answer the questions in each docstring on paper before you write the code.
2. Do not paste a solution from nanoGPT, a tutorial or an LLM. If you are
   stuck for more than 45 minutes, ask for a hint, not for the code.
3. A milestone is done when its tests pass and you can explain every line.

## Setup

    pip install -r requirements.txt
    pytest -x            # stops at the first failing test

Milestones 1 to 4 run on a laptop CPU or in a free Kaggle CPU notebook. Only
milestone 5 needs the T4.

## Milestones

| # | What you write | File | Done when |
|---|----------------|------|-----------|
| 1 | Character tokenizer and batch sampling | `minigpt/tokenizer.py` | `pytest tests/test_tokenizer.py` passes |
| 2 | Causal self-attention | `minigpt/model.py` | `pytest tests/test_model.py -k attention` passes |
| 3 | MLP and residual block | `minigpt/model.py` | `pytest tests/test_model.py -k "mlp or block"` passes |
| 4 | Full GPT: forward, loss, generate | `minigpt/model.py` | all of `pytest` passes |
| 5 | Training loop with checkpoint and resume | `train.py` | validation loss below 1.6 on Tiny Shakespeare |

At about 3 hours a week, expect one milestone every one to two weeks.

## After milestone 5

These are not scaffolded yet. Do them in this order:

6. Byte-pair encoding tokenizer, replacing the character one.
7. KV cache in `generate`; measure tokens per second before and after.
8. Mixed precision and gradient accumulation; measure memory and speed on the T4.

Milestone 8 is the bridge to the QLoRA fine-tuning project.
