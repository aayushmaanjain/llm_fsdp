"""Script to finetune LLM models."""
from datasets import load_dataset
import hydra
from omegaconf import DictConfig
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    GPT2LMHeadModel,
    DataCollatorForLanguageModeling,
    get_scheduler,
)

from trainer import Trainer


@hydra.main(config_path="../configs", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    """Main function."""
    # Only run for supported models
    assert cfg.model in set(["gpt2", "gpt2-large"]), "Model type not supported."

    # Load Model and Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(f"openai-community/{cfg.model}")
    model = GPT2LMHeadModel.from_pretrained(f"openai-community/{cfg.model}")
    print(model)
    print(f"#parameters = {model.num_parameters() / 1e6} million")

    # block_size = tokenizer.model_max_length
    block_size = int(tokenizer.model_max_length /
                     cfg.block_factor)  # TODO: change

    def tokenize_function(examples):
        return tokenizer(examples["text"])

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {
            k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported
        # it instead of this drop, you can customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i: i + block_size]
                for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    dataset = load_dataset("wikitext", "wikitext-2-v1")
    print((f"Dataset #samples: train={dataset['train'].num_rows},"
           f"val={dataset['validation'].num_rows}, test={dataset['test'].num_rows}"))
    tokenized_datasets = dataset.map(
        tokenize_function, batched=True, remove_columns=["text"])
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        desc=f"Grouping texts in chunks of {block_size}"
    )
    train_dataset = lm_datasets['train']
    val_dataset = lm_datasets['validation']
    print(f"Dataset size: train={len(train_dataset)}, val: {len(val_dataset)}")

    # Dataloaders
    # NOTE: tokenizer does not have a pad token.
    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False)
    dataloader_kwargs = {
        'batch_size': cfg.batchsize,
        'collate_fn': data_collator,
        'pin_memory': True,
    }
    train_dl = DataLoader(train_dataset, shuffle=True, **dataloader_kwargs)
    val_dl = DataLoader(val_dataset, **dataloader_kwargs)

    # Optimizer and LR Scheduler
    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=cfg.num_warmup_steps,
        num_training_steps=cfg.epochs * len(train_dl)
    )
    trainer = Trainer(model, optimizer, lr_scheduler)
    trainer.training_loop(train_dl, val_dl, cfg.epochs)


if __name__ == "__main__":
    main()
