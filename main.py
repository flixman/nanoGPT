import argparse
import logging
from pathlib import Path
import sys

from nanogpt import GPTLanguageModel, TokenizerFactory


logging.basicConfig(stream=sys.stderr, level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger("tokenizer")


def add_tokenizer_specific_arguments(parser: argparse.ArgumentParser, tokenizer_type: str) -> None:
    """Add CLI arguments contributed by the selected tokenizer implementation."""
    tokenizer_cls = TokenizerFactory.get(tokenizer_type)
    for option_name, option_kwargs in tokenizer_cls.cli_options().items():
        flag_name = f"--{tokenizer_type}:{option_name}"
        parser.add_argument(flag_name, dest=f"{tokenizer_type}_{option_name}", **option_kwargs)


def add_all_tokenizer_specific_arguments(parser: argparse.ArgumentParser) -> None:
    """Add CLI arguments contributed by every registered tokenizer implementation."""
    for tokenizer_type in TokenizerFactory.available_tokenizers():
        add_tokenizer_specific_arguments(parser, tokenizer_type)


def tokenizer_kwargs_from_args(args: argparse.Namespace) -> dict[str, object]:
    """Collect tokenizer-specific constructor arguments from the parsed CLI namespace."""
    tokenizer_cls = TokenizerFactory.get(args.tokenizer_type)
    tokenizer_kwargs: dict[str, object] = {}

    for option_name in tokenizer_cls.cli_options():
        arg_name = f"{args.tokenizer_type}_{option_name}"
        tokenizer_kwargs[option_name] = getattr(args, arg_name)

    return tokenizer_kwargs


def train_command(args: argparse.Namespace) -> None:
    from nanogpt.trainer import Trainer

    """CLI handler for model training and vocabulary serialization."""
    n_embd = args.embeddings
    n_layer = args.layers
    n_head = args.heads
    dropout = args.dropout
    block_size = args.block_size
    max_iters = args.max_iters
    eval_interval = args.eval_interval
    learning_rate = args.learning_rate
    eval_iters = args.eval_iters
    training_ratio = args.training_ratio
    training_batch_size = args.training_batch_size
    precision_bits = args.bits
    tokenizer_kwargs = tokenizer_kwargs_from_args(args)

    tokenizer = TokenizerFactory.create(
        args.tokenizer_type,
        dataset_path=Path(args.dataset),
        **tokenizer_kwargs,
    )
    tokenizer.save(Path(args.tokenizer))
    logger.info(f"Tokenizer saved to {args.tokenizer}")

    model = GPTLanguageModel(
        tokenizer.vocab_size,
        n_embd=n_embd,
        block_size=block_size,
        n_layer=n_layer,
        n_head=n_head,
        dropout=dropout,
        precision_bits=precision_bits,
    )

    ### training
    corpus_text = Path(args.dataset).read_text(encoding="utf-8")
    trainer = Trainer(model, tokenizer.encode(corpus_text), training_ratio, training_batch_size, block_size)
    trainer.train(max_iters, eval_interval, learning_rate, eval_iters)
    
    # Save the trained model
    model.save_model(Path(args.model))
    logger.info(f"Model saved to {args.model}")


def generate_command(args: argparse.Namespace) -> None:
    """CLI handler for autoregressive text generation from trained weights."""
    import random
    import time

    temperature = args.temperature
    top_k = args.top_k
    tokens = args.tokens

    tokenizer = TokenizerFactory.load(Path(args.tokenizer))
    logger.info(f"Tokenizer loaded from {args.tokenizer}")
    
    model = GPTLanguageModel.load_model(Path(args.model))
    logger.info(f"Model loaded from {args.model}")

    start_token = random.randrange(tokenizer.vocab_size)
    for idx_next in model.generate(top_k=top_k, max_new_tokens=tokens, start_token=start_token, 
                                   temperature=temperature):
        char = tokenizer.decode(idx_next[0].tolist())
        print(char, end='')
        sys.stdout.flush()
        time.sleep(0.02)


def main() -> None:
    """Parse CLI arguments and dispatch to the selected subcommand."""

    tokenizer_choices = TokenizerFactory.available_tokenizers()

    parser = argparse.ArgumentParser(description="nanoGPT training and generation")

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Training subparser
    train_parser = subparsers.add_parser("training", help="Train the model")
    train_parser.add_argument("--dropout", type=float, default=0.2, help="Dropout")
    train_parser.add_argument("--embeddings", type=int, default=256, help="Number of embeddings")
    train_parser.add_argument("--layers", type=int, default=4, help="Number of transformer layers")
    train_parser.add_argument("--heads", type=int, default=4, help="Number of heads per layer")
    train_parser.add_argument("--block_size", type=int, default=256, help="Block size")
    train_parser.add_argument("--dataset", "-d", required=True, help="Path to training dataset")
    train_parser.add_argument("--model", "-m", required=True, help="Path to save the trained model")
    train_parser.add_argument("--tokenizer", "-t", required=True, help="Path to save the tokenizer file")
    train_parser.add_argument("--tokenizer_type", "--tokenizer-type", choices=tokenizer_choices, default="char", help="Tokenizer to use")
    train_parser.add_argument("--max_iters", type=int, default=5000, help="Training iterations")
    train_parser.add_argument("--eval_interval", type=int, default=200, help="Evaluation interval")
    train_parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
    train_parser.add_argument("--eval_iters", type=int, default=200, help="Evaluation iterations")
    train_parser.add_argument("--training_ratio", type=float, default=0.9, help="Training/validation ratio")
    train_parser.add_argument("--training_batch_size", type=int, default=32, help="Training batch size")
    train_parser.add_argument("--bits", type=int, choices=(32, 16), default=32, help="Model weight precision in bits")
    add_all_tokenizer_specific_arguments(train_parser)
    train_parser.set_defaults(func=train_command)
    
    # Generation subparser
    gen_parser = subparsers.add_parser("generate", help="Generate text using a trained model")
    gen_parser.add_argument("--model", "-m", required=True, help="Path to the trained model file")
    gen_parser.add_argument("--tokenizer", "-t", required=True, help="Path to tokenizer file used for training")
    gen_parser.add_argument("--temperature", type=float, default=0.5, help="Generation temperature")
    gen_parser.add_argument("--top_k", type=int, default=50, help="Top k")
    gen_parser.add_argument("--tokens", type=int, default=1000, help="Number of tokens to print")
    gen_parser.set_defaults(func=generate_command)

    args = parser.parse_args()
    
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
