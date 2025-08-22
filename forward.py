import os 
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from gpt_oss.model import HookedGptOssForCausalLM
from sae_training.sparse_autoencoder import SparseAutoencoder
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline

torch.set_default_device(f"cuda:2")

from openai_harmony import (
    Author,
    Conversation,
    DeveloperContent,
    HarmonyEncodingName,
    Message,
    Role,
    SystemContent,
    ToolDescription,
    load_harmony_encoding,
    ReasoningEffort
)
 
encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
 
system_message = (
    SystemContent.new()
        .with_reasoning_effort(ReasoningEffort.LOW)
        .with_conversation_start_date("2025-06-28")
)
 
developer_message = (
    DeveloperContent.new()
        .with_instructions("Always respond in riddles")
)
 
convo = Conversation.from_messages(
    [
        Message.from_role_and_content(Role.SYSTEM, system_message),
        Message.from_role_and_content(Role.DEVELOPER, developer_message),
        Message.from_role_and_content(Role.USER, "What is the weather in Tokyo?"),
        Message.from_role_and_content(
            Role.ASSISTANT,
            'Need to write what the user is asking',
        ).with_channel("analysis")
    ]
)
 
tokens = encoding.render_conversation_for_completion(convo, Role.ASSISTANT)
print(tokens)

# model_id = "openai/gpt-oss-20b"

# pipe = pipeline(
#     "text-generation",
#     model=model_id,
#     torch_dtype="auto",
#     # device="cuda:2",
#     device_map="cuda:2",
# )

# messages = [
#     {"role": "user", "content": "What is the capital of France?"},
# ]

# outputs = pipe(
#     messages,
#     max_new_tokens=512,
# )
# print(outputs[0]["generated_text"][-1])


tokenizer = AutoTokenizer.from_pretrained('openai/gpt-oss-20b')
model = AutoModelForCausalLM.from_pretrained('openai/gpt-oss-20b')
# model = HookedGptOssForCausalLM.from_pretrained('openai/gpt-oss-20b')
model.eval()
model.requires_grad_(False)
model.to('cuda:2')
tokens = torch.tensor(tokens, device='cuda:2').unsqueeze(0)
output = model.generate(
    tokens, max_new_tokens=128
)

print(tokenizer.decode(output[0]))

# out = model(tokens, labels=tokens)
# print(out.loss)
# print(tokenizer.decode(tokens))
# tokens = torch.randint(100,200,size=(32,128)).to('cuda:2')

# print(out)

# tokens2 = torch.randint(100,200,size=(32,128)).to('cuda:2')
# out2 = model(tokens2)
# print(out2)