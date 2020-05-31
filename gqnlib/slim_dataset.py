
"""Dataset class for SLIM (Spatial Language Integrating Model).

ref)
https://github.com/deepmind/slim-dataset
"""

from typing import Tuple, List

import collections
import gzip
import json
import pathlib

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence


class WordVectorizer:
    """Word-vector encoder.

    ref)
    https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

    Attributes:
        word2index (collections.defaultdict): Word to index dict.
        word2count (collections.defaultdict): Word to counts dict.
        index2word (collections.defaultdict): Index to word dict.
        n_words (int): Number of words.
    """

    def __init__(self):

        self.word2index = collections.defaultdict(int)
        self.word2count = collections.defaultdict(int)
        self.index2word = {0: "UNK", 1: "SOS", 2: "EOS"}
        self.n_words = 3

        self.removal = ",.:;?!"

    def __len__(self) -> int:
        """Length of registered words."""
        return self.n_words

    def sentence2index(self, sentence: str, register: bool = True
                       ) -> List[int]:
        """Convert sentence to indices list.

        **Caution**: if `register` is `False` and unknown word is given, the
        word is registered as a key of `self.word2index` dict, but
        `self.n_words` is not incremented.

        Args:
            sentence (str): Sentence string.
            register (bool): If true, unknown words are registered to dict.

        Returns:
            indices (list of int): Indices list.
        """

        indices = []
        for word in sentence.split(" "):
            # Preprocess
            word = word.lower().strip(self.removal)

            # Register
            if register:
                if word not in self.word2index:
                    self.word2index[word] = self.n_words
                    self.word2count[word] = 1
                    self.index2word[self.n_words] = word
                    self.n_words += 1
                else:
                    self.word2count[word] += 1

            # Get index
            indices.append(self.word2index[word])

        return indices

    def to_json(self, path: str) -> None:
        """Saves to json file.

        Args:
            path (str): Path to saved file.
        """

        data = {
            "word2index": self.word2index,
            "word2count": self.word2count,
            "index2word": self.index2word,
            "n_words": self.n_words,
        }

        with open(path, "w") as f:
            json.dump(data, f)

    def read_json(self, path: str) -> None:
        """Reads saved json file.

        Args:
            path (str): Path to json file.
        """

        with open(path, "r") as f:
            data = json.load(f)

        word2index = {k: int(v) for k, v in data["word2index"].items()}
        word2count = {k: int(v) for k, v in data["word2count"].items()}

        self.word2index = collections.defaultdict(int, word2index)
        self.word2count = collections.defaultdict(int, word2count)
        self.index2word = {int(k): v for k, v in data["index2word"].items()}
        self.n_words = int(data["n_words"])

    def read_ptgz(self, path: str) -> None:
        """Reads '*.pt.gz' file.

        Args:
            path (str): Path to 'pt.gz' file.
        """

        with gzip.open(path, "rb") as f:
            dataset = torch.load(f)

        for _, _, _, cpt, *_ in dataset:
            self.sentence2index(cpt[0].decode())


class SlimDataset(torch.utils.data.Dataset):
    """SlimDataset class for SLIM.

    SlimDataset class loads data files at each time accessed by index.

    Args:
        root_dir (str): Path to root directory.
        vectorizer (WordVectorizer): Pre-trained vectorizer.

    Attributes:
        record_list (list of pathlib.Path): List of path to data files.
    """

    def __init__(self, root_dir: str, vectorizer: WordVectorizer):
        super().__init__()

        root_dir = pathlib.Path(root_dir)
        self.record_list = sorted(root_dir.glob("*.pt.gz"))
        self.vectorizer = vectorizer

    def __len__(self) -> int:
        """Returns number of files and directories in root dir.

        Returns:
            len (int): Number of objects in root dir.
        """

        return len(self.record_list)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor]:
        """Loads data file and returns data with specified index.

        This method reads `<index>.pt.gz` file, which includes a list of tuples
        `(images, viewpoints, topdown, captions, *)`.

        Args:
            index (int): Index number.

        Returns:
            images (torch.Tensor): Image tensor, size `(b, m, 3, 64, 64)`.
            viewpoints (torch.Tensor): View points, size `(b, m, 4)`.
            captions (torch.Tensor): Encoded captions, size `(b, m, l)`.
        """

        with gzip.open(self.record_list[index], "rb") as f:
            dataset = torch.load(f)

        # Read list of tuples
        images = []
        viewpoints = []
        captions = []
        for img, vwp, _, cpt, *_ in dataset:
            images.append(torch.from_numpy(img).permute(0, 3, 1, 2))
            viewpoints.append(torch.from_numpy(vwp))

            sentences = []
            for snt in cpt:
                sentences.append(torch.tensor(
                    self.vectorizer.sentence2index(snt.decode())))

            captions.append(pad_sequence(
                sentences, batch_first=False, padding_value=-1))

        images = torch.stack(images)
        viewpoints = torch.stack(viewpoints)
        captions = pad_sequence(captions, padding_value=-1).permute(1, 2, 0)

        return images, viewpoints, captions
