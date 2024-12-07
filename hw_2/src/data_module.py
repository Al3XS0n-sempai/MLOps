import torch
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule

from relu_module import MyReLU


class CustomDataset(Dataset):
    """
        My custom dataset class.
    """
    def __init__(self, data, labels):
        """
            :param data: Dataset
            :param labels: Labels
        """
        self.data = data
        self.labels = labels

    def __len__(self) -> int:
        """
            Returns length of dataset.
        """
        return len(self.data)

    def __getitem__(self, idx) -> dict:
        return {
            "features": torch.tensor(MyReLU(self.data[idx]), dtype=torch.float),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }


class CustomDataModule(LightningDataModule):
    """
        My custom data module.
    """
    def __init__(self, batch_size=32, num_workers=4):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        """
            Generate data for dataset.
        """
        self.train_data = [[i - 3000] for i in range(10000)]
        self.train_labels = [int(i > 0) for i in range(10000)]

    def setup(self, stage=None):
        if stage in (None, 'fit'):
            self.train_dataset = CustomDataset(self.train_data, self.train_labels)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
    
    def save_to_csv(self, filepath: str):
        """
            Save data to CSV.
        """
        if not hasattr(self, "train_data") or not hasattr(self, "train_labels"):
            raise ValueError("Data are not prepared prepare_data() и setup().")
        
        data = {
            "features": [item[0] for item in self.train_data],
            "label": self.train_labels
        }
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
