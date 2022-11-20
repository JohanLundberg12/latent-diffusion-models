import torch
from sklearn.metrics import f1_score

from .Trainer import Trainer
from .utils import progress_bar, timeit


class ResNetTrainer(Trainer):
    def __init__(self, config, model, train_loader, val_loader, classes):
        super().__init__(config, model, train_loader, val_loader, classes)

        self.model.to(self.device)

    @timeit
    def _train_epoch(self, epoch):
        self.model.train()

        train_loss = 0.0
        train_f1 = list()

        pbar = progress_bar(
            self.train_loader, desc=f"Train, Epoch {epoch + 1}/{self.epochs}"
        )

        for i, (data, targets) in pbar:
            data, targets = data.to(self.device), targets.to(self.device)

            # Autocasting automatically chooses the precision (floating point data type)
            # for GPU operations to improve performance while maintaining accuracy.
            with torch.cuda.amp.autocast(enabled=self.config["use_amp"]):
                outputs = self.model(data)

                loss = self.loss_fn(outputs, targets)

            # zero the parameter gradients of the optimizer
            # Make the gradients zero to avoid the gradient being a
            # combination of the old gradient and the next
            # Updates gradients by write rather than read and write (+=) used
            # https://www.youtube.com/watch?v=9mS1fIYj1So
            self.optimizer.zero_grad(set_to_none=True)

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            # update train loss and multiply by
            # data.size(0) to get the sum of the batch loss
            train_loss += loss.item() * data.size(0)

            y_preds = outputs.argmax(-1).cpu().numpy()
            y_target = targets.cpu().numpy()
            batch_f1 = f1_score(y_target, y_preds, average="micro")
            train_f1.append(batch_f1)

            # update info in progress bar
            pbar = progress_bar(
                self.train_loader,
                desc=f"Train, Epoch {epoch + 1}/{self.epochs}, train loss: {train_loss:.4f}",
            )

        # calculate average losses
        train_loss = train_loss / len(self.train_loader)
        avg_train_f1 = sum(train_f1) / len(train_f1)

        return train_loss, avg_train_f1

    # .inference_mode() should be faster than .no_grad()
    # but you can't use .requires_grad() in that context
    @timeit
    @torch.inference_mode()
    def _val_epoch(self, epoch):
        self.model.eval()

        val_loss = 0.0
        val_f1 = list()
        pbar = progress_bar(
            self.val_loader, desc=f"Val, Epoch {epoch + 1}/{self.epochs}"
        )

        for _, (data, targets) in pbar:
            data, targets = data.to(self.device), targets.to(self.device)

            outputs = self.model(data)

            loss = self.loss_fn(outputs, targets)

            val_loss += loss.item() * data.size(0)

            y_preds = outputs.argmax(-1).cpu().numpy()
            y_target = targets.cpu().numpy()
            batch_f1 = f1_score(y_target, y_preds, labels=self.classes, average="micro")
            val_f1.append(batch_f1)

            # Update info in progress bar
            pbar.set_description(
                f"Val, Epoch {epoch + 1}/{self.epochs}, val loss: {val_loss:.4f}"
            )

        # Calculate average loss and avg f1 for this epoch
        val_loss /= len(self.val_loader)
        avg_valid_f1 = sum(val_f1) / len(val_f1)

        return val_loss, avg_valid_f1

    @timeit
    def pretrain(self, dataloader):
        self.model.train()

        pretrain_loss: float = 0.0
        pretrain_f1 = list()

        pbar = progress_bar(dataloader, desc="pretrain step")

        for _, (data, targets) in pbar:
            data, targets = data.to(self.device), targets.to(self.device)

            with torch.cuda.amp.autocast():
                outputs = self.model(data)
                loss = self.loss_fn(outputs, targets)

            self.optimizer.zero_grad(set_to_none=True)

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            pretrain_loss += loss.item() * data.size(0)

            y_preds = outputs.argmax(-1).cpu().numpy()
            y_target = targets.cpu().numpy()
            batch_f1 = f1_score(y_target, y_preds, labels=self.classes, average="micro")
            pretrain_f1.append(batch_f1)

            pbar.set_description(f"pretrain step, pretrain loss: {pretrain_loss:.4f}")

        # Calculate average loss and avg f1 for this epoch
        pretrain_loss /= len(self.train_loader)
        avg_pretrain_f1 = sum(pretrain_f1) / len(pretrain_f1)

        return pretrain_loss, avg_pretrain_f1

    @timeit
    def train(self):
        """
        ### Training loop
        """
        results = {
            "train_losses": list(),
            "valid_losses": list(),
            "train_f1": list(),
            "valid_f1": list(),
        }

        for epoch in range(self.epochs):
            train_loss, train_f1 = [round(x, 4) for x in self._train_epoch(epoch)]
            valid_loss, valid_f1 = [round(x, 4) for x in self._val_epoch(epoch)]

            print(
                f"\nEpoch {epoch + 1}/{self.epochs}",
                f"\ntrain loss: {train_loss:.4f}",
                f"\ntrain f1: {train_f1:.4f}",
                f"\nvalid loss: {valid_loss:.4f}",
                f"\nvalid f1: {valid_f1:.4f}",
            )

            # Save losses
            results["train_losses"].append(train_loss)
            results["valid_losses"].append(valid_loss)
            results["train_f1"].append(train_f1)
            results["valid_f1"].append(valid_f1)

            self._log_metrics(metrics={**results}, step=epoch, mode="train")

            self.early_stopping(val_loss=valid_loss, model=self.model)
            if self.early_stopping.early_stop:
                print("Early stopping")
                break

        # divide by epoch in case of early stopping as this is should be the same as
        # len(results["train_losses"])
        avg_train_loss = sum(results["train_losses"]) / epoch
        avg_valid_loss = sum(results["valid_losses"]) / epoch
        avg_train_f1 = sum(results["train_f1"]) / epoch
        avg_valid_f1 = sum(results["valid_f1"]) / epoch

        print(
            f"\navg train-loss: {avg_train_loss}",
            f"\navg val-loss: {avg_valid_loss}",
            f"\navg train-f1: {avg_train_f1}",
            f"\navg valid-f1: {avg_valid_f1}",
        )
        print("Training done.")

    @timeit
    @torch.inference_mode()
    def predict(self, test_loader):
        print("\nPredicting on test set...")

        self.model.eval()
        f1_scores = []

        pbar = progress_bar(test_loader, desc="test step")

        for i, (data, targets) in pbar:
            data = data.to(self.device)
            output = self.model(data)

            y_preds = output.argmax(-1).cpu().numpy()
            y_target = targets.cpu().numpy()
            batch_f1 = f1_score(y_target, y_preds, labels=self.classes, average="micro")
            f1_scores.append(batch_f1)

        return f1_scores, sum(f1_scores) / len(f1_scores)
