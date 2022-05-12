import pandas as pd

from classes.trainer import Trainer


def main():
    trainer = Trainer()
    trainer.start()
    trainer.save()


if __name__ == "__main__":
    main()
