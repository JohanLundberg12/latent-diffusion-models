import torch
from sklearn.metrics import f1_score

from .Trainer import Trainer
from .utils import progress_bar, timeit


class ResNetTrainer(Trainer):
    def __init__(self, config, model, train_loader, val_loader, classes):
        super().__init__(config, model, train_loader, val_loader, classes)

        self.model.to(self.device)

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
            results = self.run(mode="train", dataloader=self.train_loader, step=epoch)
            train_loss = round(results[0], 4)
            train_f1 = round(results[1], 4)

            results = self.run(mode="valid", dataloader=self.val_loader, step=epoch)
            valid_f1 = round(results[1], 4)
            valid_loss = round(results[0], 4)

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
    def run(self, mode, dataloader, step):
        if mode == "train":
            self.model.train()

        elif mode == "val":
            self.model.eval()

        total_loss = 0.0
        f1 = list()

        pbar = progress_bar(dataloader, desc=f"{mode}, Epoch {step + 1}/{self.epochs}")

        for _, (data, targets) in pbar:
            data, targets = data.to(self.device), targets.to(self.device)

            if mode == "train":
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

            elif mode == "val":

                # .inference_mode() should be faster than .no_grad()
                # but you can't use .requires_grad() in that context
                with torch.inference_mode():
                    outputs = self.model(data)
                    loss = self.loss_fn(outputs, targets)

            # update train loss and multiply by
            # data.size(0) to get the sum of the batch loss
            total_loss += loss.item() * data.size(0)

            y_preds = outputs.argmax(-1).cpu().numpy()
            y_target = targets.cpu().numpy()
            batch_f1 = f1_score(y_target, y_preds, labels=self.classes, average="micro")
            f1.append(batch_f1)

            pbar.set_description(
                f"{mode}, Epoch {step + 1}/{self.epochs}, {mode} loss: {loss.item():.4f}, total {mode} loss: {total_loss:.4f}"
            )

        loss /= len(dataloader)
        avg_f1 = sum(f1) / len(f1)

        return loss, avg_f1
