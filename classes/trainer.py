import argparse
import torch
import torch.utils.data
import torch.nn.functional
try:
    import torch_xla.core.xla_model as xm
except Exception as e:
    xm = None
import pandas as pd
import numpy as np
from tqdm import tqdm

from classes.hate_model import HateModel
from classes.base_model import BaseModel
from classes.text_dataset import TextDataset


class Trainer():
    # Options
    _pretrained: str = 'jcblaise/roberta-tagalog-base'
    _truncation: str = 'both'
    _device: str = 'cpu'
    _tpu: bool = False
    _checkpoint: str = 'checkpoint/fake-roberta-tagalog.pt'
    _batch_size: int
    _seed: int
    _msl: int
    _wordlist: str
    wordlist: list

    _train_data: str = 'data/dataset-updated-train.csv'
    _valid_data: str = 'data/dataset-updated-valid.csv'
    _test_data: str = 'data/dataset-updated-test.csv'

    _text_column: str = 'article'
    _label_column: str = 'fake'

    epochs: int

    # Models
    _hate_model: HateModel
    _model: BaseModel
    _criterion: torch.nn
    _optimizer: torch.optim
    _scheduler: torch.optim.lr_scheduler

    def get_options(self):
        parser = argparse.ArgumentParser(description='Trainer')

        parser.add_argument('--pretrained', type=str, help='Model to be used')
        parser.add_argument('--wordlist', type=str,
                            help='Path for wordlist for negative sentiments')

        parser.add_argument('--train_data', type=str,
                            help='Path to the training dataset')
        parser.add_argument('--valid_data', type=str,
                            help='Path to the validation dataset')
        parser.add_argument('--test_data', type=str,
                            help='Path to the test dataset')
        parser.add_argument('--text_column', type=str,
                            help='Text column, eg. article')
        parser.add_argument('--label_column', type=str, help='Label Column')
        parser.add_argument('--msl', type=int, default=256,
                            help="Set the maximum sequence length to input in the tokenizer")
        parser.add_argument('--truncation', type=str,
                            help='Truncation of texts (left, right, both)', default='both')

        parser.add_argument('--lr', type=float,
                            default=1e-4, help='Learning Rate')
        parser.add_argument('--batch_size', type=int,
                            default=32, help='Batch Size')
        parser.add_argument('--epochs', type=int, default=3,
                            help='Number of epochs')
        parser.add_argument('--seed', type=int,
                            default=42, help='Seed')
        parser.add_argument('--cuda', type=bool, action='store',
                            const=True, dest='cuda', nargs='?', help='Enable GPU')
        parser.add_argument('--tpu', type=bool, action='store',
                            const=True, dest='tpu', nargs='?', help='Enable TPU')
        parser.add_argument('--checkpoint', type=str,
                            help='Where to save/load the main.')

        options = parser.parse_args()
        if options.pretrained:
            self._pretrained = options.pretrained

        if options.train_data:
            self._train_data = options.train_data
        if options.test_data:
            self._test_data = options.test_data
        if options.valid_data:
            self._valid_data = options.valid_data
        if options.text_column:
            self._text_column = options.text_column
        if options.label_column:
            self._label_column = options.label_column

        if options.cuda:
            self._device = 'cuda:0'
        elif options.tpu:
            if xm == None:
                raise 'torch_xla not installed'
            self._device = xm.xla_device()
            self._tpu = True
        if options.epochs:
            self.epochs = options.epochs
        if options.checkpoint:
            self._checkpoint = options.checkpoint

        self._wordlist = options.wordlist
        self._msl = int(options.msl)
        self._batch_size = int(options.batch_size)
        self._lr = float(options.lr)
        self._seed = int(options.seed)
        if options.truncation == 'right' or options.truncation == 'left' or options.truncation == 'both':
            self._truncation = options.truncation
        else:
            print('Invalid truncation value')

    def print_config(self):
        print('===== Configuration =====')
        print('> Pretrained:', self._pretrained)
        print('> MSL:', self._msl)
        print('> Batch Size:', self._batch_size)
        print('> Epochs:', self.epochs)
        print()

        print('===== Dataset =====')
        print('> Train:', self._train_data)
        print('> Valid:', self._valid_data)
        print('> Test:', self._test_data)
        print('> Truncation:', self._truncation)
        print('> Wordlist:', self._wordlist)
        print()

        print('===== Misc =====')
        print('> Device:', self._device)
        print('> Seed:', self._seed)
        print('> Checkpoint:', self._checkpoint)
        print()

    def __init__(self):
        self.get_options()

        torch.cuda.set_device('cuda:0' if self._tpu else self._device)

        np.random.seed(self._seed)
        torch.manual_seed(self._seed)
        torch.cuda.manual_seed(self._seed)

        # Setup
        with open(self._wordlist, 'r') as fp:
            self.wordlist = fp.read().split('\n')

        self._hate_model = HateModel(self.wordlist)

        self._model = BaseModel(
            model=self._pretrained,
            msl=self._msl,
            hate_output=self._hate_model.output,
            truncation=self._truncation,
            device=self._device
        )
        self._criterion = torch.nn.CrossEntropyLoss()
        self._optimizer = torch.optim.AdamW(
            self._model.parameters(),
            lr=self._lr,
            eps=1e-6
        )
        self._scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self._optimizer)

    def __print_eval__(self, pretext: str, loss: float, acc: float, decimal: int = 4):
        loss = round(loss, decimal)
        acc = round(acc, decimal)
        print(f'{pretext} > Loss: {loss} Acc: {acc}')

    def __load_dataset__(self, filepath: str):
        df = pd.read_csv(filepath)
        articles = df[self._text_column].to_list()

        # One hot or freq count
        hate_features = self._hate_model.frequency_count(articles)

        return TextDataset(
            articles=articles,
            hate_features=hate_features,
            labels=df[self._label_column].to_list()
        )

    def start(self):
        print('===== Preparing Datasets =====')
        print('> Setting up training dataset')
        train_data = self.__load_dataset__(self._train_data)
        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=self._batch_size, shuffle=True)

        print('> Setting up validation dataset')
        valid_data = self.__load_dataset__(self._valid_data)
        valid_loader = torch.utils.data.DataLoader(
            valid_data, batch_size=self._batch_size, shuffle=True)

        print('> Setting up test dataset')
        test_data = self.__load_dataset__(self._test_data)
        test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=self._batch_size, shuffle=True)

        print('===== Training Model ====')

        train_losses = []
        valid_losses = []
        for epoch in range(self.epochs):
            print(f'> Epoch {epoch + 1}')

            train_loss, train_acc = self.__train__(train_loader)
            train_losses.append(train_loss)

            valid_loss, valid_acc = self.__validate__(valid_loader)
            valid_losses.append(valid_loss)

            self.__print_eval__('Train', train_loss, train_acc)
            self.__print_eval__('Valid', valid_loss, valid_acc)

        eval_loss, eval_acc = self.__evaluate__(test_loader)

        self.__print_eval__('Evaluate', eval_loss, eval_acc)

    def __train__(self, loader: torch.utils.data):
        self._model.train()
        losses = []
        correct = 0
        total = 0
        for _, batch in enumerate(tqdm(loader)):
            self._optimizer.zero_grad()
            articles, hate_features, labels = batch
            hate_features = hate_features.to(self._device)
            labels = labels.to(self._device)

            output = self._model(list(articles), hate_features)

            loss = self._criterion(output, labels)
            losses.append(loss)

            pred = torch.max(output.data, 1)[1]
            total += len(labels)
            correct += int((pred == labels).sum())

            loss.backward()
            self._optimizer.step()
            if self._tpu:
                xm.mark_step()

        train_loss = sum(losses) / len(losses)
        train_acc = correct / total

        return float(train_loss), float(train_acc)

    def __validate__(self, loader: torch.utils.data):
        self._model.eval()
        losses = []
        correct = 0
        total = 0
        for _, batch in enumerate(tqdm(loader)):
            with torch.no_grad():
                articles, hate_features, labels = batch
                hate_features = hate_features.to(self._device)
                labels = labels.to(self._device)

                output = self._model(list(articles), hate_features)

                loss = self._criterion(output, labels)
                losses.append(loss)

                pred = torch.max(output.data, 1)[1]
                total += len(labels)
                correct += int((pred == labels).sum())

                del articles
                del labels

        valid_loss = sum(losses) / len(losses)
        valid_acc = correct / total
        # self._scheduler.step(valid_loss)

        return float(valid_loss), float(valid_acc)

    def __evaluate__(self, loader: torch.utils.data):
        self._model.eval()
        losses = []
        correct = 0
        total = 0
        for _, batch in enumerate(tqdm(loader)):
            with torch.no_grad():
                articles, hate_features, labels = batch
                hate_features = hate_features.to(self._device)
                labels = labels.to(self._device)

                output = self._model(list(articles), hate_features)

                loss = self._criterion(output, labels)
                losses.append(loss)

                pred = torch.max(output.data, 1)[1]
                total += len(labels)
                correct += int((pred == labels).sum())

                del articles
                del labels

        eval_loss = sum(losses) / len(losses)
        eval_acc = correct / total

        return float(eval_loss), float(eval_acc)

    def save(self):
        config = {
            'pretrained': self._pretrained,
            'msl': self._msl,
            'truncation': self._truncation,
            'wordlist': self.wordlist
        }
        torch.save([self._model.state_dict(), config, ], self._checkpoint)

    def load(self):
        state_dict, config = torch.load(self._checkpoint)
        self._pretrained = config['pretrained']
        self._msl = config['msl']
        self._truncation = config['truncation']
        self.wordlist = config['wordlist']

        self._model = BaseModel(
            model=self._pretrained,
            msl=self._msl,
            wordlist=self.wordlist,
            truncation=self._truncation,
            device=self._device
        )
        self._model.load_state_dict(state_dict)
