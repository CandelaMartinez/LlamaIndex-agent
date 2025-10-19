import sys
import transformers
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from llama_index.core import Settings
from llama_index.llms.huggingface import HuggingFaceLLM

print(sys.version)
print("Transformers OK")
print(f"Torch {torch.__version__} CUDA:{torch.cuda.is_available()}")

model_id = "sshleifer/tiny-gpt2"
tok = AutoTokenizer.from_pretrained(model_id)
mdl = AutoModelForCausalLM.from_pretrained(model_id)

llm = HuggingFaceLLM(model=mdl, tokenizer=tok, model_name=model_id, max_new_tokens=16)
Settings.llm = llm
print("LlamaIndex + HF OK")
