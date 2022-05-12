import hashlib
import os
import torch
import transformers
from tqdm import tqdm


class TransformerModel(torch.nn.Module):
    _model_str: str
    _model: transformers.AutoModel
    _tokenizer: transformers.AutoTokenizer
    _device: str

    input: int
    output: int
    truncation: str
    msl: int

    def __init__(self,
                 model: str,
                 msl: int = 0,
                 device: str = 'cpu',
                 truncation: str = 'both'
                 ):
        super(TransformerModel, self).__init__()

        self._device = device
        self.to(device)

        self._model_str = model
        if model == 'jcblaise/roberta-tagalog-large':
            self.input = 512
            self.output = 1024
        elif model == 'jcblaise/roberta-tagalog-base' or \
                model == 'jcblaise/roberta-tagalog-small' or \
                model == 'jcblaise/bert-tagalog-base-cased' or \
                model == 'jcblaise/bert-tagalog-base-uncased' or \
                model == 'jcblaise/electra-tagalog-base-cased-discriminator' or \
                model == 'jcblaise/electra-tagalog-base-uncased-discriminator':
            self.input = 512
            self.output = 768
        else:
            raise 'Invalid value for model (TransformerModel)'

        self.msl = msl if msl and msl <= self.input else self.input
        model_cache = os.path.join(
            '.transformers',
            self.__custom_hash__(model + '-model')
        )
        self._model = transformers.AutoModel.from_pretrained(
            model,
            cache_dir=model_cache
        )
        self._model.to(device)
        tokenizer_cache = os.path.join(
            '.transformers',
            self.__custom_hash__(model + '-transformer')
        )
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(
            model,
            cache_dir=tokenizer_cache
        )
        # self._tokenizer.to(device)

        if truncation == 'both':
            self.truncation = truncation
            self._tokenizer.truncation_side = 'left'
        elif truncation == 'left' or truncation == 'right':
            self.truncation = truncation
            self._tokenizer.truncation_side = truncation
        else:
            raise 'Invalid value for truncation (TransformModel)'

    def __custom_hash__(self, string: str):
        return hashlib.md5(string.encode()).hexdigest()

    def set_truncation(self, truncation: str):
        if truncation == 'both':
            self.truncation = truncation
            self._tokenizer.truncation_side = 'right'
        elif truncation == 'left' or truncation == 'right':
            self.truncation = truncation
            self._tokenizer.truncation_side = truncation
        else:
            raise 'Invalid value for truncation (TransformModel)'

    def forward(self, text: list) -> torch.Tensor:
        encodings = self._tokenizer.batch_encode_plus(
            text,
            padding=True,
            truncation=True,
            max_length=self.msl // 2 if self.truncation == 'both' else self.msl,
            return_token_type_ids=False,
            return_tensors='pt',
        )

        if self.truncation == 'both':
            self._tokenizer.truncation_side = 'left'

            encodings_2 = self._tokenizer.batch_encode_plus(
                text,
                padding=True,
                truncation=True,
                max_length=self.msl // 2,
                return_token_type_ids=False,
                return_tensors='pt',
            )
            encodings.input_ids = torch.cat(
                (encodings.input_ids, encodings_2.input_ids), dim=1)
            encodings.attention_mask = torch.cat(
                (encodings.attention_mask, encodings_2.attention_mask), dim=1)

            self._tokenizer.truncation_side = 'left'

        outputs: torch.Tensor = self._model(
            input_ids=encodings.input_ids.to(self._device),
            attention_mask=encodings.attention_mask.to(self._device)
        )
        if self._model_str == 'jcblaise/electra-tagalog-base-cased-discriminator' or \
                self._model_str == 'jcblaise/electra-tagalog-base-uncased-discriminator':
            return outputs[0].mean(1)
        return outputs[1]
