'''
Paraphrase detection for GPT starter code.

Consider:
 - ParaphraseGPT: Your implementation of the GPT-2 classification model.
 - train: Training procedure for ParaphraseGPT on the Quora paraphrase detection dataset.
 - test: Test procedure. This function generates the required files for your submission.

Running:
  `python paraphrase_detection.py --use_gpu`
trains and evaluates your ParaphraseGPT model and writes the required submission files.
'''

import argparse
import random
import torch

import numpy as np
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import (
  ParaphraseDetectionDataset,
  ParaphraseDetectionTestDataset,
  load_paraphrase_data
)
from evaluation import model_eval_paraphrase, model_test_paraphrase
from models.gpt2 import GPT2Model

from optimizer import AdamW

TQDM_DISABLE = False

# Fix the random seed.
def seed_everything(seed=11711):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True


class ParaphraseGPT(nn.Module):
  """Your GPT-2 Model designed for paraphrase detection."""

  def __init__(self, args):
    super().__init__()
    self.gpt = GPT2Model.from_pretrained(model=args.model_size, d=args.d, l=args.l, num_heads=args.num_heads)
    self.paraphrase_detection_head = nn.Linear(args.d, 2)  # Paraphrase detection has two outputs: 1 (yes) or 0 (no).


    """ lianne's edits start """
    # add dropout and head type to args
    self.dropout = nn.Dropout(args.hidden_dropout_prob)
    self.paraphrase_head_type = args.paraphrase_head_type

    # don't update GPT-2 weights if we're only fine-tuning the last linear layer
    if args.fine_tune_mode == 'last-linear-layer':
      for param in self.gpt.parameters():
        param.requires_grad = False
    # default option: fine-tune the full model
    else:
      for param in self.gpt.parameters():
        param.requires_grad = True
    """ lianne's edits end """

 
  def forward(self, input_ids, attention_mask):
    """
    TODO: Predict the label of the token using the paraphrase_detection_head Linear layer.

    We structure the input as:

      'Is "{s1}" a paraphrase of "{s2}"? Answer "yes" or "no": '

    So you want to find the prediction for the next token at the end of this sentence. Optimistically, it will be the
    token "yes" (byte pair encoding index of 8505) for examples that are paraphrases or "no" (byte pair encoding index
     of 3919) for examples that are not paraphrases.
    """

    'Takes a batch of sentences and produces embeddings for them.'
    # get hidden states from GPT-2
    gpt_output = self.gpt(input_ids, attention_mask)

    # grab the hidden state of the last token (used to make yes/no prediction)
    last_token_hidden = gpt_output['last_token']

    # cloze head: convert last token hidden state to scores for each token in vocab
    if self.paraphrase_head_type == 'cloze':
      vocab_logits = self.gpt.hidden_state_to_token(last_token_hidden)
      # extract the logits for "yes" and "no" tokens to make the prediction
      no_logit  = vocab_logits[:, 3919]
      yes_logit = vocab_logits[:, 8505]
      logits = torch.stack([no_logit, yes_logit], dim=1)
    # classifier head: directly convert last token hidden state to yes/no logits
    else:
      last_token_hidden = self.dropout(last_token_hidden)
      logits = self.paraphrase_detection_head(last_token_hidden)

    return logits


# given
def save_model(model, optimizer, args, filepath):
  save_info = {
    'model': model.state_dict(),
    'optim': optimizer.state_dict(),
    'args': args,
    'system_rng': random.getstate(),
    'numpy_rng': np.random.get_state(),
    'torch_rng': torch.random.get_rng_state(),
  }

  torch.save(save_info, filepath)
  print(f"save the model to {filepath}")


# given
def train(args):
  """Train GPT-2 for paraphrase detection on the Quora dataset."""
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  # Create the data and its corresponding datasets and dataloader.
  # para_train_data = load_paraphrase_data(args.para_train)
  # para_dev_data = load_paraphrase_data(args.para_dev)

  """ lianne's edits start """
  # my computer can't handle running the full dataset
  para_train_data = load_paraphrase_data(args.para_train)[:10000]
  para_dev_data = load_paraphrase_data(args.para_dev)[:2000]
  """ lianne's edits end """

  para_train_data = ParaphraseDetectionDataset(para_train_data, args)
  para_dev_data = ParaphraseDetectionDataset(para_dev_data, args)

  para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=args.batch_size,
                                     collate_fn=para_train_data.collate_fn)
  para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                   collate_fn=para_dev_data.collate_fn)

  args = add_arguments(args)
  model = ParaphraseGPT(args)
  model = model.to(device)

  lr = args.lr
  optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.)
  best_dev_acc = 0

  # Run for the specified number of epochs.
  for epoch in range(args.epochs):
    model.train()
    train_loss = 0
    num_batches = 0
    for batch in tqdm(para_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
      # Get the input and move it to the gpu (I do not recommend training this model on CPU).
      b_ids, b_mask, labels = batch['token_ids'], batch['attention_mask'], batch['labels'].flatten()
      b_ids = b_ids.to(device)
      b_mask = b_mask.to(device)
      labels = labels.to(device)

      """ lianne's edits start """
      # convert labels to 0/1
      labels = (labels == 8505).long()
      """ lianne's edits end """

      # Compute the loss, gradients, and update the model's parameters.
      optimizer.zero_grad()
      logits = model(b_ids, b_mask)
      preds = torch.argmax(logits, dim=1)
      loss = F.cross_entropy(logits, labels, reduction='mean')
      loss.backward()
      optimizer.step()

      train_loss += loss.item()
      num_batches += 1

    train_loss = train_loss / num_batches

    dev_acc, dev_f1, *_ = model_eval_paraphrase(para_dev_dataloader, model, device)

    # print both acc and f1
    print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, dev acc :: {dev_acc :.3f}, dev f1 :: {dev_f1 :.3f}")

    if dev_acc > best_dev_acc:
      best_dev_acc = dev_acc
      save_model(model, optimizer, args, args.filepath)


@torch.no_grad()
def test(args):
  """Evaluate your model on the dev and test datasets; save the predictions to disk."""
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  saved = torch.load(args.filepath)

  model = ParaphraseGPT(saved['args'])
  model.load_state_dict(saved['model'])
  model = model.to(device)
  model.eval()
  print(f"Loaded model to test from {args.filepath}")

  para_dev_data = load_paraphrase_data(args.para_dev)
  para_test_data = load_paraphrase_data(args.para_test, split='test')

  para_dev_data = ParaphraseDetectionDataset(para_dev_data, args)
  para_test_data = ParaphraseDetectionTestDataset(para_test_data, args)

  para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                   collate_fn=para_dev_data.collate_fn)
  para_test_dataloader = DataLoader(para_test_data, shuffle=True, batch_size=args.batch_size,
                                    collate_fn=para_test_data.collate_fn)

  dev_para_acc, dev_para_f1, dev_para_y_pred, dev_para_y_true, dev_para_sent_ids = model_eval_paraphrase(para_dev_dataloader, model, device)
  print(f"dev paraphrase acc :: {dev_para_acc :.3f}, dev paraphrase f1 :: {dev_para_f1 :.3f}")
  test_para_y_pred, test_para_sent_ids = model_test_paraphrase(para_test_dataloader, model, device)

  # write dev predictions
  with open(args.para_dev_out, "w+") as f:
    f.write("id,Predicted_Is_Paraphrase\n")
    for p, s in zip(dev_para_sent_ids, dev_para_y_pred):
      f.write(f"{p},{s}\n")

  # write test predictions
  with open(args.para_test_out, "w+") as f:
    f.write("id,Predicted_Is_Paraphrase\n")
    for p, s in zip(test_para_sent_ids, test_para_y_pred):
      f.write(f"{p},{s}\n")

  """ lianne's edits start """
  # save misclassified dev examples for error analysis
  if args.error_analysis_out:
    raw_dev = load_paraphrase_data(args.para_dev)
    id_to_example = {ex[0]: ex for ex in raw_dev}
    with open(args.error_analysis_out, "w+") as f:
      f.write("id,sentence1,sentence2,gold_label,predicted_label\n")
      for sid, pred, gold in zip(dev_para_sent_ids, dev_para_y_pred, dev_para_y_true):
        if pred != gold and sid in id_to_example:
          ex = id_to_example[sid]
          f.write(f"{sid},{ex[1]},{ex[2]},{gold},{pred}\n")
    print(f"error analysis saved to {args.error_analysis_out}")
    " lianne's edits end"


def get_args():
  parser = argparse.ArgumentParser()

  parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
  parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
  parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")
  parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
  parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")

  parser.add_argument("--seed", type=int, default=11711)
  parser.add_argument("--epochs", type=int, default=10)
  parser.add_argument("--use_gpu", action='store_true')

  parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
  parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)
  parser.add_argument("--model_size", type=str,
                      help="The model size as specified on hugging face. DO NOT use the xl model.",
                      choices=['gpt2', 'gpt2-medium', 'gpt2-large'], default='gpt2')

  """ lianne's edits start """
  # full model vs last linear layer
  parser.add_argument("--fine_tune_mode", type=str,
                      choices=['full-model', 'last-linear-layer'], default='full-model')

  # dropout value on last hidden state
  parser.add_argument("--hidden_dropout_prob", type=float, default=0.1)

  # classifier head vs cloze-style extraction
  parser.add_argument("--paraphrase_head_type", type=str,
                      choices=['classifier', 'cloze'], default='classifier')

  # error analysis output
  parser.add_argument("--error_analysis_out", type=str, default=None)
  """ lianne's edits end """

  args = parser.parse_args()
  return args


def add_arguments(args):
  """Add arguments that are deterministic on model size."""
  if args.model_size == 'gpt2':
    args.d = 768
    args.l = 12
    args.num_heads = 12
  elif args.model_size == 'gpt2-medium':
    args.d = 1024
    args.l = 24
    args.num_heads = 16
  elif args.model_size == 'gpt2-large':
    args.d = 1280
    args.l = 36
    args.num_heads = 20
  else:
    raise Exception(f'{args.model_size} is not supported.')
  return args


if __name__ == "__main__":
  args = get_args()
  # filepath name edited.
  args.filepath = f'paraphrase-{args.model_size}-{args.fine_tune_mode}-{args.epochs}e-{args.lr}.pt'
  seed_everything(args.seed)  # Fix the seed for reproducibility.
  train(args)
  test(args)
