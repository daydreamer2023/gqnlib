
import unittest

import gzip
import json
import pathlib
import tempfile

import torch

import gqnlib


class TestWordVectorizer(unittest.TestCase):

    def setUp(self):
        self.vectrizer = gqnlib.WordVectorizer()

    def test_sentence2index(self):
        sentence = "aA, ab. aa? aa! ba,.,., ba!?"
        indices = self.vectrizer.sentence2index(sentence)

        # Indices
        self.assertListEqual(indices, [3, 4, 3, 3, 5, 5])

        # Word to index
        self.assertSetEqual(set(self.vectrizer.word2index.keys()),
                            set(["aa", "ab", "ba"]))

        self.assertEqual(self.vectrizer.word2index["aa"], 3)
        self.assertEqual(self.vectrizer.word2index["ab"], 4)
        self.assertEqual(self.vectrizer.word2index["ba"], 5)

        self.assertEqual(self.vectrizer.word2index["c"], 0)
        self.assertEqual(self.vectrizer.word2index["d"], 0)

        # Word to count
        self.assertEqual(self.vectrizer.word2count["aa"], 3)
        self.assertEqual(self.vectrizer.word2count["ab"], 1)
        self.assertEqual(self.vectrizer.word2count["ba"], 2)

        self.assertEqual(self.vectrizer.word2count["c"], 0)
        self.assertEqual(self.vectrizer.word2count["d"], 0)

        # Index to word
        self.assertEqual(self.vectrizer.index2word[0], "UNK")
        self.assertEqual(self.vectrizer.index2word[3], "aa")
        self.assertEqual(self.vectrizer.index2word[4], "ab")

        # N words
        self.assertEqual(self.vectrizer.n_words, 6)

        # Length
        self.assertEqual(len(self.vectrizer), 6)

    def test_sentence2index_no_register(self):
        sentence = "aA, ab. aa? aa! ba,.,., ba!?"
        self.vectrizer.sentence2index(sentence)

        sentence = "ac, aa., BA?"
        indices = self.vectrizer.sentence2index(sentence, register=False)

        self.assertListEqual(indices, [0, 3, 5])

    def test_to_json(self):
        sentence = "aa, ab. aa? aa! ba,.,., ba!?"
        self.vectrizer.sentence2index(sentence)

        with tempfile.TemporaryDirectory() as root:
            path = pathlib.Path(root, "tmp.json")
            self.vectrizer.to_json(path)

            with path.open() as f:
                data = json.load(f)

        self.assertTrue("word2index" in data)
        self.assertTrue("word2count" in data)
        self.assertTrue("index2word" in data)
        self.assertTrue("n_words" in data)

        self.assertDictEqual(data["word2index"], {"aa": 3, "ab": 4, "ba": 5})

    def test_read_json(self):
        sentence = "aa, ab. aa? aa! ba,.,., ba!?"
        self.vectrizer.sentence2index(sentence)

        model = gqnlib.WordVectorizer()

        with tempfile.TemporaryDirectory() as root:
            path = pathlib.Path(root, "tmp.json")
            self.vectrizer.to_json(path)

            model.read_json(path)

        self.assertDictEqual(self.vectrizer.word2index, model.word2index)
        self.assertDictEqual(self.vectrizer.word2count, model.word2count)
        self.assertDictEqual(self.vectrizer.index2word, model.index2word)
        self.assertEqual(self.vectrizer.n_words, model.n_words)

    def test_read_ptgz(self):
        c = torch.randn(10, 3, 64, 64).numpy()
        cpt = [b"aa, ab. aa? aa! ba,.,., ba!?"]
        data = [(c, c, c, cpt, c, c)] * 10

        with tempfile.TemporaryDirectory() as root:
            path = pathlib.Path(root, "tmp.json")
            with gzip.open(path, "wb") as f:
                torch.save(data, f)

            self.vectrizer.read_ptgz(path)

        self.assertEqual(self.vectrizer.word2index["aa"], 3)
        self.assertEqual(self.vectrizer.word2count["aa"], 30)
        self.assertEqual(self.vectrizer.index2word[3], "aa")
        self.assertEqual(self.vectrizer.n_words, 6)


class TestSlimDataset(unittest.TestCase):

    def test_len(self):
        dataset = gqnlib.SlimDataset(".", None)
        self.assertGreaterEqual(len(dataset), 0)

    def test_getitem(self):
        # Vectorizer
        vectorizer = gqnlib.WordVectorizer()
        vectorizer.sentence2index("aa, ab. aa? aa! ba,.,., ba!?")

        # Dataset
        imgs = torch.empty(10, 64, 64, 3).numpy()
        tgts = torch.empty(10, 4).numpy()
        cpts = [b"aa, ab. aa? aa! ba,.,., ba!?"] * 10
        data = [(imgs, tgts, tgts, cpts, tgts)] * 8

        # Save and load
        with tempfile.TemporaryDirectory() as root:
            path = pathlib.Path(root, "tmp.pt.gz")
            with gzip.open(path, "wb") as f:
                torch.save(data, f)

            # Get data item
            dataset = gqnlib.SlimDataset(root, vectorizer)
            frames, viewpoints, captions = dataset[0]

        self.assertTupleEqual(frames.size(), (8, 10, 3, 64, 64))
        self.assertTupleEqual(viewpoints.size(), (8, 10, 4))
        self.assertTupleEqual(captions.size(), (8, 10, 6))


if __name__ == "__main__":
    unittest.main()
