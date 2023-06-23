from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from transformers import GPT2LMHeadModel , GPT2Tokenizer
from transformers import GPT2Model, GPT2Config
from transformers import GPT2Tokenizer
from contextlib import nullcontext
import torch.nn.init as init
import torch.nn as nn
import torch
import os
import numpy as np

class ModifiedGPT2Block(GPT2Block):
    def __init__(self, config):
        super(ModifiedGPT2Block, self).__init__(config)
        self.adapter = AdapterModule(config.n_embd, config.n_embd // 2)  # using a 2:1 ratio for the bottleneck

        # Lock all weights except adapter for training
        for name,param in self.named_parameters():
            if 'adapter' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def forward(self, hidden_states, layer_past=None, attention_mask=None, head_mask=None, use_cache=False, output_attentions=False):
        # Use the forward method of the original GPT2Block
        attn_outputs = super().forward(hidden_states, layer_past=layer_past, attention_mask=attention_mask, head_mask=head_mask, use_cache=use_cache, output_attentions=output_attentions)
        
        # Add the adapter after the multi-headed self-attention
        # Remember to add the output of the adapter to the original hidden states (residual connection)
        attn_output = attn_outputs[0]  # assuming you're only interested in the attention output
        adapter_output = self.adapter(attn_output)
        # Note this is not exactly as in the paper bcs GPT2Block internally has a residual bypass. That one should be disabled.
        hidden_states = hidden_states + adapter_output
        
        # Depending on what you're trying to achieve, you might want to modify the return statement
        return hidden_states

class AdapterModule(nn.Module):
    def __init__(self, input_size, bottleneck_size):
        super(AdapterModule, self).__init__()
        self.down_project = nn.Linear(input_size, bottleneck_size)
        self.up_project = nn.Linear(bottleneck_size, input_size)

        # Init weights close to identity
        init.uniform_(self.down_project.weight, -0.01, 0.01)
        init.uniform_(self.up_project.weight, -0.01, 0.01)
        init.uniform_(self.down_project.bias, -0.9, 1.1)
        init.uniform_(self.up_project.bias, -0.9, 1.1)

    def forward(self, x):
        down_projected = self.down_project(x)
        up_projected = self.up_project(down_projected)
        return up_projected


class ModifiedGPT2(GPT2LMHeadModel):
    def __init__(self, config):
        super(ModifiedGPT2, self).__init__(config)
        self.h = nn.ModuleList([ModifiedGPT2Block(config) for _ in range(config.n_layer)])

#if __name__ == '__main__':
## Load Data
block_size = 128
batch_size = 32
learning_rate = 1e-3
eval_interval = 10
max_iters = 100
eval_iters = 20
iter_num = 0
best_val_loss = 1e9

device = 'cpu'
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

dataset = 'shakespeare'
data_dir = os.path.join('data', dataset)
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
print('Loaded dataset...')

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                outs = model(input_ids=X, labels=Y)
                loss = outs.loss
                logits = outs.logits
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

## Load Model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2') 
model = ModifiedGPT2.from_pretrained('gpt2', pad_token_id=tokenizer.eos_token_id)
model.to(device)
print('loaded ModifiedGPT2 to device...')

## Finetune Training
# Pytorch optimizer
params = []
for name,param in model.named_parameters():
    if 'adapter' in name:
        params.append(param)
optimizer = torch.optim.AdamW(params, lr=learning_rate)

for iter in range(max_iters):
    # Evaluate loss every once in a while
    if iter % eval_interval == 0:
        losses = estimate_loss(model)
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # Sample a batch of data
    xb, yb = get_batch('train')
    # Forward
    outs = model(xb, labels=yb)
    loss = outs.loss
    logits = outs.logits
    # Backward
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    # Update
    optimizer.step()

## Inference
#tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
'''
model.eval() # Eval mode

starter_phrase = "Once upon a time"
encode = lambda msg: tokenizer.encode(msg, return_tensors='pt')
#encoded_input = tokenizer(starter_phrase, padding='longest', truncation=True, max_length=512, return_tensors='pt')
#generated_text = model.generate(encoded_input, max_length=100, temperature=0.7, do_sample=True)
generated_text = model.generate(encode(starter_phrase), max_length=100, temperature=0.7, do_sample=True)

# Decode the generated text
generated_text = tokenizer.decode(generated_text[0])

print(generated_text)
'''
