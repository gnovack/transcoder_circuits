from typing import Optional
import torch
from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssMLP, GptOssDecoderLayer, GptOssModel, GptOssForCausalLM, GptOssPreTrainedModel, GptOssRMSNorm, GptOssRotaryEmbedding, GptOssAttention, GradientCheckpointingLayer
from transformers.cache_utils import Cache
from transformers import AutoTokenizer, PreTrainedTokenizer
from transformers.integrations.mxfp4 import mlp_forward

from openai_harmony import (
    Conversation,
    DeveloperContent,
    HarmonyEncodingName,
    Message,
    Role,
    SystemContent,
    load_harmony_encoding,
    ReasoningEffort
)

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
            result, router_logits = mlp_forward(self, hidden_states)

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

        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")
        self.system_tokens = None
        self.think_tokens = None
        self.encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    
    def run_with_cache(self, input_ids, names_filter=[], stop_at_layer=None, return_type=None, loss_per_token=False):
        if names_filter:
            layer_idx = int(names_filter[0].split('.')[-1])
            self.model.enable_cache(layer_idx)

        if return_type == 'loss':
            output = self.forward(input_ids, labels=input_ids, names_filter=names_filter, stop_at_layer=stop_at_layer)

            if loss_per_token:
                loss = per_token_loss_fn(output.logits, labels=input_ids, vocab_size=self.vocab_size)
            else:
                loss = output.loss
            
            output = loss
        else:
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
                       

    
    def to_tokens(self, input, prompt=None, truncate=True, move_to_device=True):

        # input = self.tokenizer.apply_chat_template(
        #     [
        #         {"role": "user", "content": input}
        #     ],
        #     tokenize=False
        # )

        if prompt:
            tokens = self.render_convo(input, prompt)
            tokens = torch.tensor(tokens)
        else:
            tokens = self.tokenizer(
                input,
                return_tensors="pt",
                padding=True,
                truncation=truncate,
                max_length=512
            )["input_ids"]

        if move_to_device:
            tokens = tokens.to(self.device)
        return tokens

    def render_convo(self, input, prompt):
        
        if not self.system_tokens:
            system_message = (
                SystemContent.new()
                    .with_reasoning_effort(ReasoningEffort.LOW)
                    .with_conversation_start_date("2025-06-28")
            )
            system_conv = Conversation.from_messages(
                [
                    Message.from_role_and_content(Role.SYSTEM, system_message)
                ]
            )
            self.system_tokens = self.encoding.render_conversation_for_completion(system_conv, Role.USER) + [200008]
        
        if not self.think_tokens:
            think_conv = Conversation.from_messages(
                [
                    Message.from_role_and_content(
                        Role.ASSISTANT,
                        'Need to do what the user is asking.',
                    ).with_channel("analysis")
                ]
            )
            self.think_tokens = self.encoding.render_conversation_for_completion(think_conv, Role.ASSISTANT) + [200005,  17196, 200008]


        prompt_tokens = self.tokenizer(
            prompt,
            padding=False,
            truncation=True,
            add_special_tokens=False,
            max_length=512
        )["input_ids"]


        input_tokens = self.tokenizer(
            input,
            padding=True,
            truncation=True,
            add_special_tokens=False,
            max_length=512
        )["input_ids"]

        return self.system_tokens + prompt_tokens + self.think_tokens + input_tokens

def per_token_cross_entropy(
    source: torch.Tensor,
    target: torch.Tensor,
    num_items_in_batch: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    **kwargs,
) -> torch.Tensor:
    loss = torch.nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction='none')
    # if reduction == "sum":
    #     # just in case users pass an int for num_items_in_batch, which could be the case for custom trainer
    #     if torch.is_tensor(num_items_in_batch):
    #         num_items_in_batch = num_items_in_batch.to(loss.device)
    #     loss = loss / num_items_in_batch
    return loss

def per_token_loss_fn(
    logits,
    labels,
    vocab_size: int,
    num_items_in_batch: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    shift_labels: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    # Upcast to float if we need to compute the loss to avoid potential precision issues
    logits = logits.float()

    if shift_labels is None:
        # Shift so that tokens < n predict n
        labels = torch.nn.functional.pad(labels, (0, 1), value=ignore_index)
        shift_labels = labels[..., 1:].contiguous()

    # Flatten the tokens
    logits = logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(logits.device)
    loss = per_token_cross_entropy(logits, shift_labels, num_items_in_batch, ignore_index, **kwargs)
    return loss