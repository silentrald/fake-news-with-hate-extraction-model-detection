import torch
import re


class HateModel():
    _mapping: dict
    _count: dict

    output: int

    def __init__(self, wordlist: list):
        self.output = len(wordlist)

        self._mapping = {}
        self._count = {}

        wordlist.sort()
        for i, words in enumerate(wordlist):
            for word in words.split(';'):
                self._mapping[word] = i

    def __clean_text__(self, text: str) -> str:
        # Letters and dash(-)
        text = re.sub('[^a-z- ]+', '', text.strip().lower())
        return text

    def frequency_count(self, texts: list) -> torch.Tensor:
        tensor = torch.zeros([len(texts), self.output])

        for i, text in enumerate(texts):
            text = self.__clean_text__(text)
            words = text.split()
            size = len(words)
            for word in words:
                if word in self._mapping:
                    self._count[word] = self._count.get(word, 0) + 1

            for word, count in self._count.items():
                tensor[i, self._mapping[word]] += count / size

            self._count.clear()

        return tensor

    def one_hot(self, texts: list) -> torch.Tensor:
        tensor = torch.zeros([len(texts), self.output])

        for i, text in enumerate(texts):
            text = self.__clean_text__(text)
            words = text.split()

            for word in words:
                if word in self._mapping:
                    self._count[word] = 1

            for word, appeared in self._count.items():
                tensor[i, self._mapping[word]] += appeared

            self._count.clear()

        return tensor
