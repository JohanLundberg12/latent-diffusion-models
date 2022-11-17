from time import time

import torch
from sklearn.metrics import f1_score

import wandb
from src.EarlyStopping import EarlyStopping
from src.ResNetClassifier import ResNetBase
from src.utils import progress_bar


class ResNetTrainer:
    def __init__(
        self,
        config: dict,
        model: ResNetBase,
        train_loader,
        val_loader,
        loss_fn,
        optimizer,
        scaler,
        classes,
        device,
        epochs,
    ) -> None:
        self.config = config
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scaler = scaler
        self.classes = classes
        self.epochs = epochs
        self.device = device
        self.early_stopping = EarlyStopping(
            patience=10,
            verbose=True,
            path=self.config.data_paths["checkpoints"] + "/checkpoint.pt",
        )

    def train_step(self, epoch):
        self.model.train()

        train_loss = 0.0

        train_f1 = list()

        start_time = time()

        pbar = progress_bar(self.train_loader, desc="train step")

        for _, (data, targets) in pbar:
            # prepare data
            data, targets = data.to(self.device), targets.to(self.device)
            prepare_time = start_time - time()

            # Autocasting automatically chooses the precision (floating point data type)
            # for GPU operations to improve performance while maintaining accuracy.
            with torch.cuda.amp.autocast():
                outputs = self.model(data)

                loss = self.loss_fn(outputs, targets)

            # Updates gradients by write rather than read and write (+=) used
            # https://www.youtube.com/watch?v=9mS1fIYj1So
            self.optimizer.zero_grad(set_to_none=True)

            # The network loss is scaled by a scaling factor to prevent underflow.
            # Gradients flowing back through the network are scaled by the same factor.
            # Calls .backward() on scaled loss to create scaled gradients.
            self.scaler.scale(loss).backward()

            # Scaler.step() first unscales the gradients of the optimizer's
            # assigned params by dividing them by the scale factor.
            # If the gradients do not contain NaNs/inf, optimizer.step() is called,
            # otherwise skipped.
            # optimizer.step() is then called using the unscaled gradients.
            self.scaler.step(self.optimizer)

            # Updates the scale factor
            self.scaler.update()

            # Add total batch loss to total loss
            batch_loss = loss.item() * data.size(0)  # not sure about data.size(0)
            train_loss += batch_loss

            y_preds = outputs.argmax(-1).cpu().numpy()
            y_target = targets.cpu().numpy()
            batch_f1 = f1_score(y_target, y_preds, labels=self.classes, average="micro")
            train_f1.append(batch_f1)

            # Update info in progress bar
            process_time = start_time - time() - prepare_time
            compute_efficency = process_time / (process_time + prepare_time)
            pbar.set_description(
                "Train step, "
                f"compute efficiency: {compute_efficency:.2f}, "
                f"batch loss: {batch_loss:.4f}, "
                f"train loss: {train_loss:.4f}, "
                f"batch f1: {batch_f1:.4f}, "
            )
            start_time = time()

        # Calculate average loss and avg f1 for this epoch
        train_loss /= len(self.train_loader)
        avg_train_f1 = sum(train_f1) / len(train_f1)

        return train_loss, avg_train_f1

    # .inference_mode() should be faster than .no_grad()
    # but you can't use .requires_grad() in that context
    @torch.inference_mode()
    def val_step(self, epoch):
        self.model.eval()

        valid_loss: float = 0.0

        valid_f1 = list()

        start_time = time()

        pbar = progress_bar(self.val_loader, desc="val step")

        for _, (data, targets) in pbar:
            data, targets = data.to(self.device), targets.to(self.device)
            prepare_time = start_time - time()

            outputs = self.model(data)

            # Calc. loss
            loss = self.loss_fn(outputs, targets)

            # * data.size(0) to get total loss for the batch and not the avg.
            batch_loss = loss.item() * data.size(0)
            valid_loss += batch_loss

            y_preds = outputs.argmax(-1).cpu().numpy()
            y_target = targets.cpu().numpy()
            batch_f1 = f1_score(y_target, y_preds, labels=self.classes, average="micro")
            valid_f1.append(batch_f1)

            # Update info in progress bar
            process_time = start_time - time() - prepare_time
            compute_efficency = process_time / (process_time + prepare_time)
            pbar.set_description(
                "Val step, "
                f"compute efficiency: {compute_efficency:.2f}, "
                f"batch loss: {batch_loss:.4f}, "
                f"valid loss: {valid_loss:.4f}, "
                f"batch f1: {batch_f1:.4f}, "
            )
            start_time = time()

        # Calculate average loss and avg f1 for this epoch
        valid_loss /= len(self.val_loader)
        avg_valid_f1 = sum(valid_f1) / len(valid_f1)

        return valid_loss, avg_valid_f1

    def pretrain(self, dataloader):
        self.model.train()

        pretrain_loss: float = 0.0
        pretrain_f1 = list()

        start_time = time()

        pbar = progress_bar(dataloader, desc="pretrain step")

        for _, (data, targets) in pbar:
            data, targets = data.to(self.device), targets.to(self.device)
            prepare_time = start_time - time()

            with torch.cuda.amp.autocast():
                outputs = self.model(data)
                loss = self.loss_fn(outputs, targets)

            self.optimizer.zero_grad(set_to_none=True)

            self.scaler.scale(loss).backward()

            self.scaler.step(self.optimizer)

            self.scaler.update()

            # Add total batch loss to total loss
            batch_loss = loss.item() * data.size(0)  # not sure about data.size(0)
            pretrain_loss += batch_loss

            y_preds = outputs.argmax(-1).cpu().numpy()
            y_target = targets.cpu().numpy()
            batch_f1 = f1_score(y_target, y_preds, labels=self.classes, average="micro")
            pretrain_f1.append(batch_f1)

            process_time = start_time - time() - prepare_time
            compute_efficency = process_time / (process_time + prepare_time)
            pbar.set_description(
                "Pretrain step, "
                f"compute efficiency: {compute_efficency:.2f}, "
                f"batch loss: {batch_loss:.4f}, "
                f"train loss: {pretrain_loss:.4f}, "
                f"batch f1: {batch_f1:.4f}, "
            )
            start_time = time()

        # Calculate average loss and avg f1 for this epoch
        pretrain_loss /= len(self.train_loader)
        avg_pretrain_f1 = sum(pretrain_f1) / len(pretrain_f1)

        return pretrain_loss, avg_pretrain_f1

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

        for epoch in range(1, self.epochs + 1):
            start = time()
            train_loss, train_f1 = [round(x, 4) for x in self.train_step(epoch)]
            valid_loss, valid_f1 = [round(x, 4) for x in self.val_step(epoch)]
            stop = time()

            print(
                f"\nEpoch: {epoch+1}",
                f"\navg train-loss: {train_loss}",
                f"\navg val-loss: {valid_loss}",
                f"\navg train-f1: {train_f1}",
                f"\navg valid-f1: {valid_f1}",
                f"\ntime: {stop-start:.4f}\n",
            )

            # Save losses
            results["train_losses"].append(train_loss)
            results["valid_losses"].append(valid_loss)
            results["train_f1"].append(train_f1)
            results["valid_f1"].append(valid_f1)

            # Log results to wandb
            wandb.log({"train_loss": train_loss, "epoch": epoch})
            wandb.log({"valid_loss": valid_loss, "epoch": epoch})
            wandb.log({"train_f1": train_f1, "epoch": epoch})
            wandb.log({"valid_f1": valid_f1, "epoch": epoch})

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

        wandb.log({"train_time": stop - start})
        print("Training done.")

    @torch.inference_mode()
    def predict(self, test_loader):
        print("Predicting on test set...")

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
