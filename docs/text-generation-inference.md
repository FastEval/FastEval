**Important**: For most models you do not need to follow these instructions at all.
It is better to use vLLM if it is supported for your model.
Only use text-generation-inference if your model is not supported by vLLM or you encounter other problems with vLLM.

[Text-generation-inference](https://github.com/huggingface/text-generation-inference) is an inference server from huggingface.
It consists of a Rust based webserver that controls a number of model shards written in python.
While it is more designed as a server to provide clients on a different computer with access to a model,
we use it here as a subprocess and communicate with it on the same computer to evaluate a model.

Text-generation-inference is more complex than vLLM to install and it also takes a longer time (~30 minutes+) due to the need to compile flash-attention.
To install, follow these instructions:

```bash
# Install various system packages.
# The following command assumes an ubuntu >= 22.04 system.
# For other systems, see https://github.com/huggingface/text-generation-inference#local-install
apt install protobuf-compiler libssl-dev gcc pkg-config g++ make

# Install rust.
# The rust-all package on ubuntu >= 22.04 might also work, but sometimes a newer version is required.
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install text-generation-inference to the `FastEval/text-generation-inference` folder.
# This will compile flash-attention which can take a long time depending on the number of CPU cores you have.
./install-text-generation-inference
```
