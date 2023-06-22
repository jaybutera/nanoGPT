from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from transformers import GPT2LMHeadModel , GPT2Tokenizer
from transformers import GPT2Model, GPT2Config
from transformers import GPT2Tokenizer
import torch.nn.init as init
import torch.nn as nn

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
# Usage
tokenizer = GPT2Tokenizer.from_pretrained('gpt2') 
#config = GPT2Config.from_pretrained('gpt2', pad_token_id=tokenizer.eos_token_id)
#model = GPT2LMHeadModel.from_pretrained('gpt2', pad_token_id=tokenizer.eos_token_id)
#model = ModifiedGPT2(config)
model = ModifiedGPT2.from_pretrained('gpt2', pad_token_id=tokenizer.eos_token_id)

## Finetune Training
#optim = torch.optim.Adam

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
