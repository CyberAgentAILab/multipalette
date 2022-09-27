import os
import random
from collections import Counter

import numpy as np
import tensorflow as tf
from model_config import Config


class Tokenizer:
    def __init__(self, config):
        with open(config["vocabulary_file_path"], "r", encoding="utf-8") as f:
            self.dict = ["SEP", "MASK", "PAD", "UNK"] + eval(f.read())
        self.word2id = {self.dict[i]: i for i in range(len(self.dict))}
        self.id2word = {i: self.dict[i] for i in range(len(self.dict))}

        self._token_end_id = self.word2id["SEP"]
        self._token_mask_id = self.word2id["MASK"]
        self._token_pad_id = self.word2id["PAD"]
        self._token_unknown_id = self.word2id["UNK"]

    def encode(self, text):
        #         token_ids = [self._token_start_id] + [self.word2id[char] for char in text] + [self._token_end_id]
        token_ids = [self.word2id[char] for char in text]
        segment_ids = [0 for char in text]
        return token_ids, segment_ids

    def decode(self, ids):
        return self.id2word[ids]


class Corpus:
    def __init__(self, config):
        self.config = config
        self.vocab2id, self.id2vocab = self.generate_vocabulary()
        self.data = []

    def generate_vocabulary(self):

        if os.path.exists(self.config["vocabulary_file_path"]):
            with open(self.config["vocabulary_file_path"], "r", encoding="utf-8") as f:
                vocabs = eval(f.read())
        else:
            with open(self.config["corpus_file_path"], "r", encoding="utf-8") as f:
                corpus_ = f.read()
            vocabs_with_frequency = Counter(corpus_).most_common()
            vocabs = [
                word for (word, freq) in vocabs_with_frequency if freq > self.config["character_frequency_threshold"]
            ]
            with open(self.config["vocabulary_file_path"], "w", encoding="utf-8") as f:
                f.write(str(vocabs))

        vocabs = ["SEP", "MASK", "PAD", "UNK"] + vocabs
        vocab2id = dict(zip(vocabs, list(range(len(vocabs)))))
        id2vocab = dict(zip(list(range(len(vocabs))), vocabs))

        #         print('Vocabulary Size = {}'.format(len(vocab2id)))

        return vocab2id, id2vocab

    def make_and_parse_passages(self):
        with open(self.config["corpus_file_path"], "r", encoding="utf-8") as f:
            corpus_ = f.readlines()
        for line in corpus_:
            yield line.replace('"', "")

    def make_bert_data(self):
        passages = self.make_and_parse_passages()
        for passage in passages:
            sentences = passage.strip("\n").split(" ; ")
            if len(sentences) == 1:
                print("1 palette only")
                continue
            one_sample = []
            for i in range(len(sentences)):
                for color in sentences[i].split(" "):
                    if color == "":
                        one_sample.append(self.vocab2id["PAD"])
                    else:
                        if color in self.vocab2id:
                            one_sample.append(self.vocab2id[color])
                        else:
                            one_sample.append(self.vocab2id["UNK"])
                # add PAD when color number in a palette is less then max_palette_length
                for r in range(len(sentences[i].split(" ")), self.config["max_palette_length"][i]):
                    one_sample.append(self.vocab2id["PAD"])
                one_sample.append(self.vocab2id["SEP"])

            if len(one_sample) < self.config["max_sequence_length"]:
                one_sample += [self.vocab2id["PAD"]] * (self.config["max_sequence_length"] - len(one_sample))
            self.data.append(one_sample[: self.config["max_sequence_length"]])

    def token_id_to_word_list(self, token_id_list):
        """
        transfer token_id to original word list
        """
        word_list = []
        for token_id in token_id_list:
            if token_id in self.id2vocab:
                word_list.append(self.id2vocab[token_id])
            else:
                word_list.append("[UNK]")
        return word_list


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, config):
        self.config = config
        self.corpus = Corpus(config)
        self.corpus.make_bert_data()
        self.data = self.corpus.data
        self.batch_size = self.config["batch_size"]
        #         assert self.batch_size % 2 == 0, 'ensure batch_size be an even number for paired data: sample, neg_sample'
        self.mask_token_id = self.corpus.vocab2id["MASK"]

    def __len__(self):
        return len(self.data) // self.batch_size

    def make_mask_language_model_data(self, batch_token_id):
        """
        Real mask rate for token ids is mask_rate * mask_token_rate.
        """
        batch_size = len(batch_token_id)
        batch_ignorePAD = (np.array(batch_token_id) != self.corpus.vocab2id["PAD"]).astype(int)
        batch_ignoreSEP = (np.array(batch_token_id) != self.corpus.vocab2id["SEP"]).astype(int)
        batch_ignore = (batch_ignorePAD * batch_ignoreSEP).astype(int)
        # print(batch_ignore)
        batch_real_seq_lens = np.sum(batch_ignore, axis=1)
        batch_mask_word_num = np.ceil(batch_real_seq_lens * self.config["mask_rate"]).astype(int)

        mask_position = []
        for i in range(batch_size):
            real_seq = [idx for idx, element in enumerate(batch_ignore[i]) if element > 0]
            if len(self.config["mask_position"]) == 0:  # set random mask position
                prob = random.random()
                if prob < Config["mask_token_rate"]:
                    position = np.random.choice(
                        real_seq, size=batch_mask_word_num[i], replace=False
                    )  # set random position
                else:
                    position = []
            else:
                position = self.config["mask_position"]  # set fixed mask position
            mask_position.append(np.sum(np.eye(self.config["max_sequence_length"])[position], axis=0))

        mask_position = np.array(mask_position)
        # set masked position with mask token id
        mask_value_matrix = mask_position * self.mask_token_id
        inputs_mask = (mask_position == 0).astype(int)
        batch_token_id_after_mlm = (batch_token_id * inputs_mask + mask_value_matrix).astype(int)

        # set masked position with its original token id
        inputs_unmask = (mask_position == 1).astype(int)
        mask_classification = (batch_token_id * inputs_unmask).astype(int)

        return batch_token_id_after_mlm, mask_position, mask_classification

    def make_segment_inputs(self, batch_token_id):
        segment_inputs = []

        for i in range(len(batch_token_id)):
            # fixed segmentation
            segment_input = []
            for seg in range(self.config["segment_size"]):
                for pi in range(self.config["max_palette_length"][seg] + 1):
                    segment_input.append(seg)
            segment_inputs.append(segment_input)
        segment_inputs = np.array(segment_inputs)
        return segment_inputs

    def make_padding_mask(self, batch_token_id):
        batch_padding_mask = (np.array(batch_token_id) == self.corpus.vocab2id["PAD"]).astype(int)
        return batch_padding_mask

    def __getitem__(self, idx):
        batch_data = self.data[idx * self.batch_size : (idx + 1) * self.batch_size]

        batch_x, batch_mlm_mask, batch_mcc_mask = self.make_mask_language_model_data(batch_data)
        segment_x = self.make_segment_inputs(batch_data)
        padding_mask = self.make_padding_mask(batch_data)
        shuffle = np.random.choice(np.arange(self.batch_size), size=self.batch_size, replace=False)
        batch_x, batch_segment, batch_padding_mask = batch_x[shuffle], segment_x[shuffle], padding_mask[shuffle]
        origin_x, batch_mlm_mask, batch_mcc_mask = (
            np.array(batch_data)[shuffle],
            batch_mlm_mask[shuffle],
            batch_mcc_mask[shuffle],
        )

        # return original tokens, masked sentences, masked positions, original sentences, segments, padding positions, pos-neg labels
        return batch_x, batch_mlm_mask, batch_mcc_mask, origin_x, batch_segment, batch_padding_mask


if __name__ == "__main__":
    dataset = DataGenerator(Config)
    batch_x, batch_mlm_mask, batch_mcc_mask, origin_x, batch_segment, batch_padding_mask = dataset[0]
    print(f"original sequence: {dataset.corpus.token_id_to_word_list(list(origin_x[0]))}")
    print(f"original id: {origin_x[0]}")
    print(f"segment: {batch_segment[0]}")
    #     print(f'[PAD] mask: {batch_padding_mask[0]}')
    print(f"masked sequence: {dataset.corpus.token_id_to_word_list(list(batch_x[0]))}")
    print(f"batch_x: {batch_x[0]}")
    print(f"mcc_mask: {batch_mcc_mask[0]}")
    print(f"mlm_mask: {batch_mlm_mask[0]}")
    print(f"pad_mask: {batch_padding_mask[0]}")
