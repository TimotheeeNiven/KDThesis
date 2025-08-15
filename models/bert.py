from __future__ import absolute_import

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

__all__ = ['roberta', 'distilbert', 'tinybert', 'mobilebert', 'albert', 'bertbase', 'bertsmall']

class BERT_QA(nn.Module):
    def __init__(self, model_name="bert-base-uncased", task="qa", return_features=True):
        super(BERT_QA, self).__init__()
        self.task = task
        self.return_features = return_features

        # Load config without pretrained weights
        self.config = AutoConfig.from_pretrained(model_name)
        self.config.output_hidden_states = True
        self.encoder = AutoModel.from_pretrained(model_name, config=self.config)


        # Token type support
        self.supports_token_type = getattr(self.config, "type_vocab_size", 0) > 1

        # Task head
        if self.task == "qa":
            self.head = nn.Linear(self.config.hidden_size, 2)
        elif self.task == "cls":
            self.head = nn.Linear(self.config.hidden_size, self.config.num_labels)
        else:
            raise ValueError(f"Unsupported task: {self.task}")

        self.head.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, is_feat=False):
        kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if self.supports_token_type:
            kwargs["token_type_ids"] = token_type_ids if token_type_ids is not None else torch.zeros_like(input_ids)

        outputs = self.encoder(**kwargs)
        hidden_states = outputs.hidden_states
        sequence_output = outputs.last_hidden_state
        logits = self.head(sequence_output)

        if self.task == "qa":
            start_logits, end_logits = logits.split(1, dim=-1)
            start_logits = start_logits.squeeze(-1)
            end_logits = end_logits.squeeze(-1)
            return (hidden_states, (start_logits, end_logits)) if (is_feat or self.return_features) else (start_logits, end_logits)
        else:
            cls_logits = logits[:, 0, :]
            return (hidden_states, cls_logits) if (is_feat or self.return_features) else cls_logits


# Factory methods
def roberta(): return BERT_QA("roberta-base")
def distilbert(): return BERT_QA("distilbert-base-uncased")
def tinybert(): return BERT_QA("prajjwal1/bert-tiny")
def mobilebert(): return BERT_QA("google/mobilebert-uncased")  # <== Re-added here
def albert(): return BERT_QA("albert-base-v2")
def bertbase(): return BERT_QA("bert-large-uncased")
def bertsmall(): return BERT_QA("bert-base-uncased")


# Quick test
if __name__ == '__main__':
    model = mobilebert()
    x = torch.randint(0, model.encoder.config.vocab_size, (2, 384))
    mask = torch.ones_like(x)
    token_types = torch.zeros_like(x)
    feats, (start, end) = model(x, mask, token_types, is_feat=True)
    print("Shapes:", start.shape, end.shape, "Features:", feats[-1].shape)
