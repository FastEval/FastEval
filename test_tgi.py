#!/usr/bin/env python3

# export PATH=$PATH:/workspace/ilm-eval/text-generation-inference/target/release
# export USE_FLASH_ATTENTION=TRUE
# text-generation-launcher --model-id OpenAssistant/falcon-7b-sft-mix-2000 --max-total-tokens 4096 --max-input-length 2048 --hostname 127.0.0.1 --port 1234 --huggingface-hub-cache /workspace/huggingface-cache --dtype b-float16

from text_generation import Client

input_text = "<|prompter|>Can you tell me what is brutalism?<|endoftext|><|assistant|>"

client = Client("http://127.0.0.1:1234")
print(client.generate(input_text, max_new_tokens=64).generated_text)

text = ""
for response in client.generate_stream(input_text, max_new_tokens=64):
    if not response.token.special:
        text += response.token.text
print(text)
