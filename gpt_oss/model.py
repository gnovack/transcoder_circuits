from typing import List, Literal, Optional, Union
import numpy as np
import torch
from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssMLP, GptOssDecoderLayer, GptOssModel, GptOssForCausalLM, GptOssPreTrainedModel, GptOssRMSNorm, GptOssRotaryEmbedding, GptOssAttention, GradientCheckpointingLayer, repeat_kv
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

        if self.replacement_hook:
            result = self.replacement_hook(hidden_states)
        else:
            result, router_logits = mlp_forward(self, hidden_states)

        if self.cache:
            self.act_cache[f'mlp_input.{self.layer_idx}'] = hidden_states
            self.act_cache[f'mlp_output.{self.layer_idx}'] = result
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
        self.layer_idx = layer_idx

    def OV(self, head):
        
        start = head*self.self_attn.head_dim
        end = start+self.self_attn.head_dim
        
        v = self.self_attn.v_proj.weight[:,None,:].expand(512,self.self_attn.num_key_value_groups,2880).reshape(4096,2880)
        o = self.self_attn.o_proj.weight
        
        return torch.matmul(
            v[start:end,:].transpose(0,1), o[:,start:end].transpose(0,1)
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs,
    ) -> tuple[torch.Tensor]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        if self.mlp.cache:
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, self.self_attn.head_dim)
            value_states = self.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            value_states = repeat_kv(value_states, self.self_attn.num_key_value_groups)
            self.mlp.act_cache[f'value.{self.layer_idx}'] = value_states

            self.mlp.act_cache[f'residual.{self.layer_idx}'] = residual
        
        # Self Attention
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        if self.mlp.cache:
            self.mlp.act_cache[f'attn_pattern.{self.layer_idx}'] = attn_weights
            self.mlp.act_cache[f'mlp_input_pre_norm.{self.layer_idx}'] = hidden_states

        hidden_states, _ = self.mlp(hidden_states)  # diff with llama: router scores
        hidden_states = residual + hidden_states
        return hidden_states

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
        else:
            for i in range(len(self.model.layers)):
                self.model.enable_cache(i)


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
        else:
            caches = self.model.pop_caches()
            for i in range(len(self.model.layers)):
                self.model.disable_cache(i)
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
                       
    
    def to_str_tokens(
        self,
        input: Union[
            str,
            torch.Tensor,
            np.ndarray,
            list,
        ],
        prepend_bos: Optional[Union[bool, None]] = None,
        padding_side: Optional[Union[Literal["left", "right"], None]] = None,
    ) -> Union[List[str], List[List[str]]]:
        """Map text, a list of text or tokens to a list of tokens as strings.

        Gotcha: prepend_bos prepends a beginning of string token. This is a recommended default when
        inputting a prompt to the model as the first token is often treated weirdly, but should only
        be done at the START of the prompt. If prepend_bos=None is passed, it implies the usage of
        self.cfg.default_prepend_bos which is set to True unless specified otherwise. Therefore,
        make sure to locally turn it off by passing prepend_bos=False if you're looking at the
        tokenization of part of the prompt! (Note: some models eg GPT-2 were not trained with a BOS
        token, others (OPT and my models) were)

        Gotcha2: Tokenization of a string depends on whether there is a preceding space and whether
        the first letter is capitalized. It's easy to shoot yourself in the foot here if you're not
        careful!

        Gotcha3: If passing a string that exceeds the model's context length (model.cfg.n_ctx), it
        will be truncated.

        Args:
            input (Union[str, list, torch.Tensor]): The input - either a string or a tensor of
                tokens. If tokens, should be a tensor of shape [pos] or [1, pos].
            prepend_bos (bool, optional): Overrides self.cfg.default_prepend_bos. Whether to prepend
                the BOS token to the input (only applies when input is a string). Defaults to None,
                implying usage of self.cfg.default_prepend_bos which is set to True unless specified
                otherwise. Pass True or False to locally override the default.
            padding_side (Union[Literal["left", "right"], None], optional): Overrides
                self.tokenizer.padding_side. Specifies which side to pad when tokenizing multiple
                strings of different lengths.

        Returns:
            str_tokens: List of individual tokens as strings
        """
        assert self.tokenizer is not None  # keep mypy happy
        tokens: Union[np.ndarray, torch.Tensor]
        if isinstance(input, list):
            return list(
                map(
                    lambda tokens: self.to_str_tokens(tokens, prepend_bos, padding_side),
                    input,
                )
            )  # type: ignore
        elif isinstance(input, str):
            tokens = self.to_tokens(input)[
                0
            ]
            # Gemma tokenizer expects a batch dimension
            if "gemma" in self.tokenizer.name_or_path and tokens.ndim == 1:
                tokens = tokens.unsqueeze(1)
        elif isinstance(input, torch.Tensor):
            tokens = input
            tokens = tokens.squeeze()  # Get rid of a trivial batch dimension
            if tokens.dim() == 0:
                # Don't pass dimensionless tensor
                tokens = tokens.unsqueeze(0)
            assert (
                tokens.dim() == 1
            ), f"Invalid tokens input to to_str_tokens, has shape: {tokens.shape}"
        elif isinstance(input, np.ndarray):
            tokens = input
            tokens = tokens.squeeze()  # Get rid of a trivial batch dimension
            if tokens.ndim == 0:
                # Don't pass dimensionless tensor
                tokens = np.expand_dims(tokens, axis=0)
            assert (
                tokens.ndim == 1
            ), f"Invalid tokens input to to_str_tokens, has shape: {tokens.shape}"
        else:
            raise ValueError(f"Invalid input type to to_str_tokens: {type(input)}")
        str_tokens = self.tokenizer.batch_decode(tokens, clean_up_tokenization_spaces=False)
        return str_tokens
    
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

