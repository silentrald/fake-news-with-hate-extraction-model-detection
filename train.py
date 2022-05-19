from classes.trainer import Trainer


def main():
    trainer = Trainer()
    trainer.print_config()
    trainer.start()
    trainer.save()


if __name__ == "__main__":
    main()
