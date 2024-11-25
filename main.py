from lightning.pytorch.cli import LightningCLI
from modules.module import Module
from modules.datamodule import DataModule


def main() -> None:
    LightningCLI(
        model_class=Module,
        datamodule_class=DataModule,
        save_config_kwargs={"overwrite": True}
    )


if __name__ == '__main__':
    main()
