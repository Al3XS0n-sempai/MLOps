import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from src.data_module import CustomDataModule
from src.model_module import SimpleModel


@hydra.main(version_base="1.3", config_path="./configs", config_name="simple_config")
def main(cfg: DictConfig):
    data_module = CustomDataModule(
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers
    )

    model = SimpleModel(input_dim=cfg.model.input_dim, lr=cfg.train.lr)

    trainer = Trainer(max_epochs=cfg.train.epochs)
    trainer.fit(model, data_module)

    data_module.save_to_csv(filepath=f"./data/dataset_{cfg.data.version}.csv")


if __name__ == "__main__":
    main()