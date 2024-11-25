import torch

from constants.networkenum import Network
from factories.networkfactory import NetworkFactory
from torch.optim.lr_scheduler import StepLR
from torch.optim import Adam
from torch.nn.functional import cross_entropy
from torchmetrics import Accuracy
from lightning.pytorch import LightningModule
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class Module(LightningModule):

    def __init__(
            self,
            num_classes: int,
            learning_rate: float,
            network: Network,
            pretrained: bool
        ) -> None:
        super(Module, self).__init__()
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.save_hyperparameters()
        self.model = NetworkFactory.create(ntype=network, num_classes=num_classes, pretrained=pretrained)
        self.train_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.train_losses = list()
        self.val_losses = list()

    def forward(self, tensors: Tensor) -> Tensor:
        output = self.model(tensors)
        return output

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> dict[str, Tensor]:
        values, labels = batch
        outputs = self(values)
        loss = cross_entropy(outputs, labels)
        self.train_losses.append(loss.detach())
        acc = self.train_acc(outputs, labels)
        self.log_dict({
            'train_loss': loss.detach(),
            'train_acc': acc
        }, prog_bar=True)
        return {'loss': loss}

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> None:
        values, labels = batch
        outputs = self(values)
        loss = cross_entropy(outputs, labels)
        self.val_losses.append(loss)
        acc = self.val_acc(outputs, labels)
        self.log_dict({
            'val_loss': loss,
            'val_acc': acc
        }, prog_bar=True)

    def on_train_epoch_end(self) -> None:
        mean_loss = torch.stack(self.train_losses).mean()
        mean_acc = self.train_acc.compute()
        self.log_dict({
            'train_epoch_loss': mean_loss,
            'train_epoch_acc': mean_acc}
        )
        self.train_losses.clear()

    def on_validation_epoch_end(self) -> None:
        mean_loss = torch.stack(self.val_losses).mean()
        mean_acc = self.val_acc.compute()
        self.log_dict({
            'val_epoch_loss': mean_loss,
            'val_epoch_acc': mean_acc}
        )
        self.val_losses.clear()

    def configure_optimizers(self) -> tuple[list[Optimizer], list[LRScheduler]]:
        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        scheduler = StepLR(optimizer, step_size=50, gamma=0.1)
        return [optimizer], [scheduler]
