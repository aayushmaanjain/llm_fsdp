"""Module providing code for training models."""
from typing import Any, Dict, Optional, Union

import evaluate
import torch
from torch.utils.data import DataLoader
import wandb

from utils import timer

class Trainer:
    """Trainer"""
    def __init__(self,
                 model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
                 ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.metrics: Dict[str, Any] = {}
        self.print_freq = 100

        self.step = 0

    def training_loop(self,
                      train_dl: DataLoader,
                      val_dl: DataLoader,
                      epochs: int
                      ):
        """Training Loop that runs for desired number of epochs."""
        # num_training_steps = epochs * len(train_dl)
        # progress_bar = tqdm(range(num_training_steps))
        self.model = self.model.to(self.device)
        for i in range(epochs):
            train_metrics = self.train(train_dl, i+1)
            print(f"Epoch {i+1}")
            print("Train", train_metrics)
            val_metrics = self.evaluate(val_dl)
            print("Validation:", val_metrics)
            wandb.log({"train": train_metrics, "val": val_metrics, "epoch": i+1})

    @timer
    def train(self, train_dl: DataLoader, epoch: int = 0):
        """Trains model for one epoch."""
        train_loss = torch.tensor(0.).to(self.device)
        self.model.train()
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
            wandb.log({"train": {"batch_loss": loss.item()}}, step=self.step)
            self.step += 1
            if step % self.print_freq == 0:
                print(f"Epoch {epoch:02}::{step}/{len(train_dl)}: Loss{loss.item()}")
        train_loss /= len(train_dl)
        return {'loss': train_loss.item()}

    @timer
    def evaluate(self, val_dl: DataLoader):
        """Evaluate."""
        accmetric = evaluate.load("accuracy", module_type="metric")
        # TODO add perplexity and other metrics?
        # perpmetric = evaluate.load("perplexity", module_type="metric")
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
        return {
            "accuracy": accmetric.compute()['accuracy'],
            "loss": val_loss.item(),
        }
