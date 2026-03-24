# nanoGPT

A compact PyTorch implementation of a GPT-style language model for text generation. The project includes a training CLI, multiple tokenizer options, and a saved model/tokenizer pair you can use to generate text right away.

## What is included

- Transformer-based autoregressive language model in PyTorch
- Training and generation commands in `main.py`
- Multiple tokenizer backends: character-level, BPE, and tiktoken-based
- Reference notes in `docs/` for both the neural-network foundations and the GPT architecture

## Project layout

- `main.py` - command-line entry point for training and generation
- `nanogpt/` - model, training loop, and tokenizer implementations
- `docs/` - technical background and mathematical notes

## Requirements

Install the Python dependencies listed in `requirements.txt`.

## Usage

### Train a model

To train a model with all Shakespeare's works:

```bash
mkdir input
curl -s https://ocw.mit.edu/ans7870/6/6.006/s08/lecturenotes/files/t8.shakespeare.txt | \
     sed -E 's/\b[A-Z]+\b//g' | tail -n +241 | sed -E "s/^\s\s*//" > input/shakespeare.txt

python main.py training \
  --dataset input/shakespeare.txt \
  --model trained_model.pt \
  --tokenizer tokenizer.json \
  --tokenizer_type bpe
```

Tokenizer-specific training options are exposed with a `--<tokenizer>:<option>` prefix.

### Generate text

```bash
python main.py generate \
  --model trained_model.pt \
  --tokenizer tokenizer.json \
  --tokens 200
```

## Documentation

- [docs/GPT_README.md](docs/GPT_README.md) for the GPT architecture and training overview
- [docs/NeuralNetwork_README.md](docs/NeuralNetwork_README.md) for the underlying neural-network math

## Notes

This started as an example used in a tutorial from [Gabriel Merlo](https://www.youtube.com/watch?v=QK4AHZTVf68). The initial code can be found [here](https://gabriels-organization-67.gitbook.io/construyamos-gpt).
