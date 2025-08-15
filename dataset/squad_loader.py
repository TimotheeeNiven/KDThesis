from __future__ import print_function

import os
import socket
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast
from transformers import AutoTokenizer

def get_data_folder():
    """
    Return server-dependent path to store the data
    """
    hostname = socket.gethostname()
    if hostname.startswith('visiongpu'):
        data_folder = '/data/vision/phillipi/rep-learn/datasets'
    elif hostname.startswith('yonglong-home'):
        data_folder = '/home/yonglong/Data/data'
    else:
        data_folder = './data/'

    os.makedirs(data_folder, exist_ok=True)
    return data_folder


class SquadDataset(Dataset):
    """
    SQuAD v1.1 Dataset for BERT QA fine-tuning
    """
    def __init__(self, json_path, tokenizer, max_length=384, doc_stride=128):
        with open(json_path, 'r') as f:
            squad = json.load(f)

        self.samples = []
        for article in squad['data']:
            for paragraph in article['paragraphs']:
                context = paragraph['context']
                for qa in paragraph['qas']:
                    question = qa['question']
                    qid = qa['id']
                    answers = qa['answers']

                    if not answers:
                        answers = [{'text': '', 'answer_start': 0}]

                    answer = answers[0]
                    answer_text = answer['text']
                    answer_start = answer['answer_start']
                    answer_end = answer_start + len(answer_text)

                    tokenized = tokenizer(
                        question,
                        context,
                        truncation="only_second",
                        max_length=max_length,
                        stride=doc_stride,
                        padding="max_length",
                        return_offsets_mapping=True,
                        return_tensors="pt"
                    )

                    for i in range(tokenized['input_ids'].size(0)):
                        inputs = {k: v[i] for k, v in tokenized.items()}
                        inputs['start_positions'] = 0
                        inputs['end_positions'] = 0
                        offset_mapping = inputs['offset_mapping']
                        sequence_ids = tokenized.sequence_ids(i)

                        if answer_text:
                            token_start_index = next((i for i, s in enumerate(sequence_ids) if s == 1), 0)
                            token_end_index = max((i for i, s in enumerate(sequence_ids) if s == 1), default=0)

                            for idx in range(token_start_index, token_end_index + 1):
                                start, end = offset_mapping[idx].tolist()
                                if start <= answer_start < end:
                                    inputs['start_positions'] = idx
                                if start < answer_end <= end:
                                    inputs['end_positions'] = idx

                        inputs['offset_mapping'] = offset_mapping.tolist()  # ? Keep this
                        inputs['context'] = context                       # ? Add context
                        inputs['id'] = qid
                        inputs['answers'] = {'text': [answer_text], 'answer_start': [answer_start]}
                        self.samples.append(inputs)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def squad_collate_fn(batch):
    out_batch = {}

    for key in batch[0]:
        values = [item[key] for item in batch]

        # Skip stacking list/dict (like answers or metadata)
        if isinstance(values[0], (dict, list, str)):
            out_batch[key] = values
            continue

        # Convert all elements to tensors with explicit dtype
        tensor_values = []
        for v in values:
            if isinstance(v, torch.Tensor):
                tensor_values.append(v)
            elif isinstance(v, (int, float)):
                tensor_values.append(torch.tensor(v, dtype=torch.long))
            else:
                raise TypeError(f"squad_collate_fn: Unexpected type {type(v)} for key '{key}'")

        out_batch[key] = torch.stack(tensor_values)

    return out_batch

def get_squad_dataloaders(batch_size=16, num_workers=4, tokenizer_name='bert-base-uncased'):
    data_folder = get_data_folder()
    squad_folder = os.path.join(data_folder, 'SQuAD')
    train_file = os.path.join(squad_folder, 'train-v1.1.json')
    val_file = os.path.join(squad_folder, 'dev-v1.1.json')

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    train_set = SquadDataset(train_file, tokenizer)
    val_set = SquadDataset(val_file, tokenizer)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=squad_collate_fn
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=squad_collate_fn
    )

    return train_loader, val_loader
