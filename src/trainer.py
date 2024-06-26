"""Module providing code for training models."""
from typing import Optional

import evaluate
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import wandb

from utils import timer, free_cuda_memory

class Trainer:
    """Trainer"""

    def __init__(self,
                 model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
                 rank: int = 0,
                 world_size: int = 1
                 ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = torch.device(f"cuda:{torch.cuda.current_device()}") \
                      if torch.cuda.is_available() else torch.device("cpu")
        self.rank = rank
        self.world_size = world_size
        self.print_freq = 100
        self.step = 0

    @timer
    def training_loop(self,
                      train_dl: DataLoader,
                      val_dl: DataLoader,
                      test_dl: Optional[DataLoader] = None,
                      epochs: int = 1
                      ):
        """Training Loop that runs for desired number of epochs."""
        # Evaluate pretrained model on validation set to set baseline.
        val_metrics = self.evaluate(val_dl)
        if self.rank == 0:
            print("Validation baseline:", val_metrics)
            wandb.log({"val": val_metrics, "epoch": 0}, commit=False)
        # Run training
        for i in range(epochs):
            train_metrics = self.train(train_dl, i+1)
            # NOTE: Done to avoid getting OOM during validation.
            # TODO: investigate why OOM is happening.
            free_cuda_memory()
            if self.rank == 0:
                print(f"Epoch {i+1}")
                print("Train", train_metrics)
            val_metrics = self.evaluate(val_dl)
            # NOTE: Done to avoid getting OOM during training in the next epoch.
            # TODO: investigate why OOM is happening.
            free_cuda_memory()
            if self.rank == 0:
                print("Validation:", val_metrics)
                wandb.log({"train": train_metrics, "val": val_metrics, "epoch": i+1})

        # Evaluate on test set
        test_metrics = {}
        if test_dl is not None:
            test_metrics.update(self.evaluate(test_dl))
            if self.rank == 0:
                print("Test:", test_metrics)
                for k, v in test_metrics.items():
                    wandb.run.summary[f"test_{k}"] = v

        # Save final model
        if self.world_size > 1:
            dist.barrier()
        states = self.model.state_dict()
        if self.rank == 0:
            torch.save(states, "gpt2-finetuned.pt")

        return test_metrics

    @timer
    def train(self, train_dl: DataLoader, epoch: int = 0):
        """Trains model for one epoch."""
        train_loss = torch.tensor(0.).to(self.device)
        self.model.train()
        # Set epoch for distributed sampler
        if isinstance(train_dl.sampler, DistributedSampler):
            train_dl.sampler.set_epoch(epoch)
        # Iterate over batches.
        for step, batch in enumerate(train_dl, start=1):
            self.optimizer.zero_grad()
            batch = {k: v.to(self.device) for k, v in batch.items()}
            out = self.model(**batch)
            loss = out.loss
            train_loss += loss.item()
            loss.backward()
            self.optimizer.step()
            if self.lr_scheduler:
                self.lr_scheduler.step()
            # Log and update progress
            if self.rank == 0:
                wandb.log({"train": {"batch_loss": loss.item()}}, step=self.step)
            self.step += 1
            if self.rank == 0 and step % self.print_freq == 0:
                print(f"Epoch {epoch:02}::{step}/{len(train_dl)}: Loss{loss.item()}")
        train_loss /= len(train_dl)
        if self.world_size > 1:
            dist.all_reduce(train_loss, op=dist.ReduceOp.AVG)
        return {'loss': train_loss.item()}

    @timer
    def evaluate(self, val_dl: DataLoader):
        """Evaluate."""
        # TODO: distributed evaluate metrics? It seems like evaluate implicitly handles
        # distributed evaluation. Revisit later to confirm.
        accmetric = evaluate.load("accuracy", module_type="metric")
        val_loss = torch.tensor(0.).to(self.device)
        self.model.eval()
        with torch.no_grad():
            for batch in val_dl:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                out = self.model(**batch)
                val_loss += out.loss.item()
                predictions = torch.argmax(out.logits, dim=-1)
                accmetric.add_batch(
                    predictions=torch.flatten(predictions.type(torch.int32)),
                    references=torch.flatten(batch["labels"].type(torch.int32))
                )
        val_loss /= len(val_dl)
        if self.world_size > 1:
            dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        try:
            perplexity = torch.exp(val_loss)
        except OverflowError:
            perplexity = torch.tensor(float('inf'))
        return {
            "accuracy": accmetric.compute()['accuracy'],
            "loss": val_loss.item(),
            "perplexity": perplexity.item(),
        }
