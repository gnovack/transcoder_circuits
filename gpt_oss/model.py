from typing import Optional
import torch
from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssMLP, GptOssDecoderLayer, GptOssModel, GptOssForCausalLM, GptOssPreTrainedModel, GptOssRMSNorm, GptOssRotaryEmbedding, GptOssAttention, GradientCheckpointingLayer
from transformers.cache_utils import Cache
from transformers import AutoTokenizer
from transformers.integrations.mxfp4 import mlp_forward

class HookedGptOssMlp(GptOssMLP):

    def __init__(self, config, layer_idx):
        super().__init__(config)
        # self.__class__.__name__ = "GptOssMLP"
        self.layer_idx = layer_idx
        self.cache = False
        self.act_cache = {}

        self.replacement_hook = None
    
    def forward(self, hidden_states):
        # router_scores, router_indices = self.router(hidden_states)  # (num_experts, seq_len)
        # routed_out = self.experts(hidden_states, router_indices=router_indices, routing_weights=router_scores)
        # result = super().forward(hidden_states)

        if self.replacement_hook:
            result = self.replacement_hook(hidden_states)
        else:
            result, _ = mlp_forward(self, hidden_states)

        if self.cache:
            self.act_cache = {
                f'mlp_input.{self.layer_idx}': hidden_states,
                f'mlp_output.{self.layer_idx}': result,
            }
        return result, None

class HookedGptOssDecoderLayer(GptOssDecoderLayer):
    def __init__(self, config: GptOssConfig, layer_idx: int):
        super(GptOssDecoderLayer, self).__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = GptOssAttention(config=config, layer_idx=layer_idx)
        self.mlp = HookedGptOssMlp(config, layer_idx)
        self.input_layernorm = GptOssRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GptOssRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention_type = config.layer_types[layer_idx]
        
        # super().__init__(config, layer_idx)
        # del self.mlp
        # self.mlp = HookedGptOssMlp(config, layer_idx)

class HookedGptOssModel(GptOssModel):

    def __init__(self, config):
        super(GptOssModel, self).__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = torch.nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = torch.nn.ModuleList(
            [HookedGptOssDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = GptOssRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = GptOssRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()
    
    def enable_cache(self, layer_idx):
        self.layers[layer_idx].mlp.cache = True
    
    def disable_cache(self, layer_idx):
        self.layers[layer_idx].mlp.cache = False
    
    def set_hook(self, layer_idx, hook):
        self.layers[layer_idx].mlp.replacement_hook = hook
    
    def unset_hook(self, layer_idx):
        self.layers[layer_idx].mlp.replacement_hook = None
    
    def pop_caches(self):
        caches = {}
        for l in self.layers:
            for k,v in l.mlp.act_cache.items():
                caches[k] = v
            l.mlp.act_cache = {}
        return caches

class HookedGptOssForCausalLM(GptOssForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = HookedGptOssModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.num_local_experts
        self.num_experts_per_tok = config.num_experts_per_tok

        # Initialize weights and apply final processing
        self.post_init()

        self.tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")
    
    def run_with_cache(self, input_ids, names_filter=[], stop_at_layer=None):
        if names_filter:
            layer_idx = int(names_filter[0].split('.')[-1])
            self.model.enable_cache(layer_idx)

        output = self.forward(input_ids, names_filter=names_filter, stop_at_layer=stop_at_layer)

        caches = {}
        if names_filter:
            caches = self.model.pop_caches()
            layer_idx = int(names_filter[0].split('.')[-1])
            self.model.disable_cache(layer_idx)
        return output, caches

    def run_with_hooks(self, input_ids, return_type, fwd_hooks):
        if return_type == 'loss':
            
            assert len(fwd_hooks) == 1, "Only supports one hook at a time for now!"

            layer_idx = int(fwd_hooks[0][0].split('.')[-1])
            self.model.set_hook(layer_idx, fwd_hooks[0][1])

            output = self.forward(input_ids, labels=input_ids)
            
            self.model.unset_hook(layer_idx)
            
            return output.loss
        else:
            raise NotImplementedError("Only 'loss' is implemented!")
                       

    
    def to_tokens(self, input, truncate=True, move_to_device=True):

        tokens = self.tokenizer(
            input,
            return_tensors="pt",
            padding=True,
            truncation=truncate,
            max_length=131_072
        )["input_ids"]

        if move_to_device:
            tokens = tokens.to(self.device)
        return tokens