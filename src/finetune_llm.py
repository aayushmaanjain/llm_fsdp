"""Script to finetune LLM models."""
from functools import partial
import os
import warnings

from datasets import load_dataset
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import torch.multiprocessing as mp
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import (
    AutoTokenizer,
    GPT2LMHeadModel,
    DataCollatorForLanguageModeling,
    get_scheduler,
)
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
import wandb

from trainer import Trainer


def setup(rank: int, world_size: int):
    """Initialize distributed group."""
    os.environ['MASTER_ADDR'] = "localhost"
    os.environ['MASTER_PORT'] = "12355"
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


def cleanup():
    """Close distributed group."""
    dist.destroy_process_group()


def fsdp_main(rank: int, world_size: int, cfg: DictConfig):
    """Function to fine-tune LLMs that supports FSDP as well."""
    if world_size > 1:
        setup(rank, world_size)

    if rank == 0:
        # Initialize wandb
        wandb.init(
            project="llm-fsdp",
            entity="aayushmaan",
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        wandb.define_metric("epoch")
        wandb.config.world_size = world_size

    # Load Model and Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    model = GPT2LMHeadModel.from_pretrained(cfg.model)
    # Enable gradient checkpointing if specified.
    if cfg.enable_gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    if rank == 0:
        print(model)
        print(f"#parameters = {model.num_parameters() / 1e6} million")
        wandb.run.summary["parameters"] = model.num_parameters()

    block_size = int(tokenizer.model_max_length / cfg.block_factor)

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

    # TODO: extend cfg specification to all datasets (beyond wikitext).
    # Will need to handle tuple/ str cases.
    dataset = load_dataset("wikitext", cfg.wikitext_dataset)
    if rank == 0:
        print((
            f"Dataset #samples: train={dataset['train'].num_rows},"
            f"val={dataset['validation'].num_rows}, test={dataset['test'].num_rows}"))

    # Preprocess the dataset
    tokenized_datasets = dataset.map(
        tokenize_function, batched=True, remove_columns=["text"])
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        desc=f"Grouping texts in chunks of {block_size}"
    )
    train_dataset = lm_datasets['train']
    val_dataset = lm_datasets['validation']
    test_dataset = lm_datasets['test']
    if rank == 0:
        print((f"Dataset size: train={len(train_dataset)}, "
               f"val: {len(val_dataset)}, test: {len(test_dataset)}"))

    # Dataloaders
    # NOTE: tokenizer does not have a pad token.
    tokenizer.pad_token = tokenizer.eos_token
    if world_size > 1:
        # We shuffle in the sampler instead in the distributed case.
        train_sampler = DistributedSampler(
            train_dataset, rank=rank, num_replicas=world_size, shuffle=True)
        val_sampler = DistributedSampler(
            val_dataset, rank=rank, num_replicas=world_size)
        test_sampler = DistributedSampler(
            test_dataset, rank=rank, num_replicas=world_size)
        per_gpu_batchsize = cfg.batchsize // world_size
    else:
        train_sampler, val_sampler, test_sampler = None, None, None
        per_gpu_batchsize = cfg.batchsize
    # Update effective batchsize.
    if cfg.batchsize % world_size != 0:
        warnings.warn((
            f"Batchsize {cfg.batchsize} cannot be divided equally between {world_size} processes."
            f"Effective batchsize = {per_gpu_batchsize * world_size}"))
        if rank == 0:
            wandb.config.batch_size = per_gpu_batchsize * world_size
            wandb.config.per_gpu_batchsize = per_gpu_batchsize

    dataloader_kwargs = {
        'batch_size': per_gpu_batchsize,
        'collate_fn': DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        'pin_memory': True,
    }
    train_dl = DataLoader(train_dataset, shuffle=(train_sampler is None),
                          sampler=train_sampler, **dataloader_kwargs)
    val_dl = DataLoader(val_dataset, sampler=val_sampler, **dataloader_kwargs)
    test_dl = DataLoader(test_dataset, sampler=test_sampler, **dataloader_kwargs)
    total_train_steps = cfg.epochs * len(train_dl)
    if rank == 0:
        wandb.run.summary["total_train_steps"] = total_train_steps

    # Send model to device.
    torch.cuda.set_device(rank)
    # FSDP model
    if world_size > 1:
        gpt2_auto_wrap_policy = partial(
            transformer_auto_wrap_policy, transformer_layer_cls={GPT2Block})
        model = FSDP(model, auto_wrap_policy=gpt2_auto_wrap_policy,
                     device_id=torch.cuda.current_device())
    else:
        model = model.to("cuda")

    # Optimizer and LR Scheduler
    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate,
                      weight_decay=cfg.weight_decay)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=cfg.num_warmup_steps,
        num_training_steps=total_train_steps
    )
    trainer = Trainer(model, optimizer, lr_scheduler, rank, world_size)
    results = trainer.training_loop(train_dl, val_dl, test_dl, cfg.epochs)
    if rank == 0:
        print("Time taken for training loop:", results["time"])

    # Cleanup in the distributed case.
    if world_size > 1:
        cleanup()


@hydra.main(config_path="../configs", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    """Main function."""
    # Only run for supported models
    assert cfg.model in set(["gpt2", "gpt2-large"]), "Model type not supported."

    # Distributed Training
    # NOTE: currently auto-scales to number of available GPUs.
    num_gpus = torch.cuda.device_count()
    if cfg.enable_fsdp and num_gpus > 1:
        print(f"Running FSDP on {num_gpus} GPUs.")
        mp.spawn(fsdp_main, args=(num_gpus, cfg), nprocs=num_gpus, join=True)
    else:
        fsdp_main(0, 1, cfg)


if __name__ == "__main__":
    main()
