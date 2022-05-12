import torch

from classes.transformer_model import TransformerModel
from classes.hate_model import HateModel


class BaseModel(torch.nn.Module):
    _pretrained: str
    _device: str
    _hate_model: HateModel
    _transformer: TransformerModel
    _linear: torch.nn.Sequential
    input: int
    output: int = 2

    def __init__(self,
                 model: str,
                 msl: int,
                 hate_output: int,
                 truncation: str = 'both',
                 device: str = 'cpu'):
        super(BaseModel, self).__init__()

        self._pretrained = model
        self._device = device

        self._transformer = TransformerModel(
            model=model,
            msl=msl,
            truncation=truncation,
            device=device
        )
        self.input = self._transformer.output + hate_output

        #########
        # Linear Layer Setup
        #########
        linear = torch.nn.Sequential()
        linear.to(device)
        self.to(device)

        # Roberta Models
        if model == 'jcblaise/roberta-tagalog-large' or \
                model == 'jcblaise/roberta-tagalog-base' or \
                model == 'jcblaise/roberta-tagalog-small':
            d1 = torch.nn.Dropout(p=0.25).to(device)
            d2 = torch.nn.Dropout(p=0.25).to(device)
            # relu = torch.nn.LeakyReLU().to(device)
            relu = torch.nn.Tanh().to(device)
            # sig = torch.nn.Sigmoid().to(device)

            ll1 = torch.nn.Linear(self.input, self.input).to(device)
            torch.nn.init.kaiming_uniform_(ll1.weight, nonlinearity='relu')
            ll2 = torch.nn.Linear(self.input, self.output).to(device)
            torch.nn.init.kaiming_uniform_(ll2.weight, nonlinearity='relu')

            linear.add_module('d1', d1)
            linear.add_module('ll1', ll1)
            linear.add_module('r', relu)
            # linear.add_module('sig', sig)
            linear.add_module('d2', d2)
            linear.add_module('ll2', ll2)

        # Bert/Electra Models
        else:
            d = torch.nn.Dropout(p=0.25).to(device)
            ll = torch.nn.Linear(self.input, self.output).to(device)

            linear.add_module('d', d)
            linear.add_module('ll', ll)

        softmax = torch.nn.Softmax(dim=1).to(device)
        linear.add_module('sm', softmax)

        self._linear = linear

    def forward(self, texts: list, hate_features: torch.Tensor) -> torch.Tensor:
        word_embeddings = self._transformer(texts)

        combined_features = torch.concat(
            [word_embeddings, hate_features],
            dim=1
        ).to(self._device)

        predicted: torch.Tensor = self._linear(combined_features)
        return predicted
