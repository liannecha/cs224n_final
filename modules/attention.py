import torch

from einops import rearrange
from torch import nn


class CausalSelfAttention(nn.Module):
  def __init__(self, config):
    super().__init__()

    self.num_attention_heads = config.num_attention_heads
    self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
    self.all_head_size = self.num_attention_heads * self.attention_head_size

    # Initialize the linear transformation layers for key, value, query.
    self.query = nn.Linear(config.hidden_size, self.all_head_size)
    self.key = nn.Linear(config.hidden_size, self.all_head_size)
    self.value = nn.Linear(config.hidden_size, self.all_head_size)
    # This dropout is applied to normalized attention scores following the original
    # implementation of transformer. Although it is a bit unusual, we empirically
    # observe that it yields better performance.
    self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

  def transform(self, x, linear_layer):
    # The corresponding linear_layer of k, v, q are used to project the hidden_state (x).
    proj = linear_layer(x)
    # Next, we need to produce multiple heads for the proj. This is done by spliting the
    # hidden state to self.num_attention_heads, each of size self.attention_head_size.
    proj = rearrange(proj, 'b t (h d) -> b t h d', h=self.num_attention_heads)
    # By proper transpose, we have proj of size [bs, num_attention_heads, seq_len, attention_head_size].
    proj = rearrange(proj, 'b t h d -> b h t d')
    return proj

  def attention(self, key, query, value, attention_mask):
    """
    Goal: calculate the multi-head attention using softmax(QK^T / sqrt(d))V.
    Inputs: batch size, number of heads, sequence length, head size
    key: [bs, num_attention_heads, seq_len, attention_head_size]
    query: [bs, num_attention_heads, seq_len, attention_head_size]
    value: [bs, num_attention_heads, seq_len, attention_head_size]
    attention_mask: [bs, 1, 1, seq_len] ??
    return: [bs, seq_len, hidden_state] ??
    """
    # extract dim of Q/K/V
    bs, num_attention_heads, seq_len, attention_head_size = query.size()

    # calculate attention scores; scaled by sqrt(d) for stability
    attn_scores = torch.matmul(query, key.transpose(-1, -2)) / (d ** 0.5)

    # causal mask (prevent access to future tokens)
    i = torch.arange(seq_len).unsqueeze(1)  # shape [seq_len, 1]
    j = torch.arange(seq_len).unsqueeze(0)  # shape [1, seq_len]

    causal_mask = j > i

    # mask out future tokens with -inf (so that after softmax they become 0)
    attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))

    # Add (padding) attention mask if provided (broadcasts over heads and query positions)
    if attention_mask is not None:
      attn_scores = attn_scores + attention_mask

    # Normalize to probabilities
    attn_probs = torch.softmax(attn_scores, dim=-1)
    attn_probs = self.dropout(attn_probs)

    # Weighted sum of values: [bs, h, t, d]
    context = torch.matmul(attn_probs, value)

    # Merge heads back: [bs, t, h*d] = [bs, seq_len, hidden_size]
    context = rearrange(context, 'b h t d -> b t (h d)')
    return context


  def forward(self, hidden_states, attention_mask):
    """
    hidden_states: [bs, seq_len, hidden_state]
    attention_mask: [bs, 1, 1, seq_len]
    output: [bs, seq_len, hidden_state]
    """
    # First, we have to generate the key, value, query for each token for multi-head attention
    # using self.transform (more details inside the function).
    # Size of *_layer is [bs, num_attention_heads, seq_len, attention_head_size].
    key_layer = self.transform(hidden_states, self.key)
    value_layer = self.transform(hidden_states, self.value)
    query_layer = self.transform(hidden_states, self.query)
    
    # Calculate the multi-head attention.
    attn_value = self.attention(key_layer, query_layer, value_layer, attention_mask)
    return attn_value
