import torch
from sklearn.metrics import f1_score

from .Trainer import Trainer
from .utils import progress_bar, timeit


class ResNetTrainer(Trainer):
    def __init__(self, config, model, train_loader, val_loader, classes):
        super().__init__(config, model, train_loader, val_loader, classes)

        self.model.to(self.device)

    @timeit
    def train(self, exp_name=None):
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
            result = self.run(
                mode="train", dataloader=self.train_loader, step=epoch + 1
            )
            train_loss = round(result[0].item(), 4)
            train_f1 = round(result[1].item(), 4)

            result = self.run(mode="valid", dataloader=self.val_loader, step=epoch + 1)
            valid_f1 = round(result[1].item(), 4)
            valid_loss = round(result[0].item(), 4)

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

            self._log_metrics(
                metrics={
                    f"{exp_name} train_loss": train_loss,
                    f"{exp_name} train_f1": train_f1,
                },
                step=epoch,
            )
            self._log_metrics(
                metrics={
                    f"{exp_name} valid_loss": valid_loss,
                    f"{exp_name} valid_f1": valid_f1,
                },
                step=epoch,
            )

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
        if mode == "train" or mode == "pretrain":
            self.model.train()

        elif mode == "valid" or mode == "test":
            self.model.eval()

        else:
            raise ValueError(f"mode {mode} not supported")

        total_loss = 0.0
        f1 = list()

        pbar = progress_bar(dataloader, desc=f"{mode}, Epoch {str(step)}/{self.epochs}")

        for _, (data, targets) in pbar:
            data, targets = data.to(self.device), targets.to(self.device)

            if mode == "train" or mode == "pretrain":
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

            elif mode == "valid":
                # The scaler is not necessary during evaluation,
                # as you won’t call backward in this step and
                # thus there are not gradients to scale.
                with torch.cuda.amp.autocast(enabled=self.config["use_amp"]):
                    # .inference_mode() should be faster than .no_grad()
                    # but you can't use .requires_grad() in that context
                    with torch.inference_mode():
                        outputs = self.model(data)
                        loss = self.loss_fn(outputs, targets)

            elif mode == "test":
                with torch.cuda.amp.autocast(enabled=self.config["use_amp"]):
                    with torch.inference_mode():
                        outputs = self.model(data)

            batch_f1 = f1_score(
                targets.cpu().numpy(),
                outputs.argmax(-1).cpu().numpy(),
                average="micro",
            )
            f1.append(batch_f1)

            if mode == "train" or mode == "pretrain" or mode == "valid":
                # update train loss and multiply by
                # data.size(0) to get the sum of the batch loss
                total_loss += loss.item() * data.size(0)
                pbar.set_description(
                    f"{mode}, Epoch {step}/{self.epochs}, Total Loss: {total_loss:.4f} F1: {batch_f1:.4f}"
                )

            elif mode == "test":
                pbar.set_description(
                    f"{mode}, Epoch {step}/{self.epochs}, F1: {batch_f1:.4f}"
                )

        total_loss /= len(dataloader)
        avg_f1 = sum(f1) / len(f1)

        if mode == "train" or mode == "pretrain" or mode == "valid":
            return loss, avg_f1
        elif mode == "test":
            return f1, avg_f1
