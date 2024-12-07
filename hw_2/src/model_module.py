import torch
from torch import nn
from pytorch_lightning import LightningModule


class SimpleModel(LightningModule):
    def __init__(self, input_dim: int, lr: float):
        super().__init__()
        self.save_hyperparameters()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 10),
            nn.Softmax(1),
            nn.Linear(10, 2)
        )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch["features"], batch["label"]
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)