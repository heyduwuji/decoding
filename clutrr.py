from transformers import pipeline
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers.pipelines.text_generation import TextGenerationPipeline
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from transformers.models.llama.modeling_llama import LlamaModel, LlamaForCausalLM
from tqdm import tqdm
from myUtils import forward

model_id = "/data1/zhaoed/models/meta-llama/Meta-Llama-3-8B-Instruct"
batch_size = 8

pipe = pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
    batch_size=batch_size,
)

prompt = {"role": "system", "content": "You are a helpful assistant. You will receive a query. Please directly output the answer starting with \"The answer is\" without any prefix."}
# prompt = {"role": "system", "content": "You are a helpful assistant. You will receive a query. Please think step-by-step and output each step of thingking, finally output the answer in a new line in a format \"So the answer is [answer]\", while [answer] should be only one word."}

# messages = [
#     # {"role": "system", "content": "You are a caculator, accept an equation and return the result only in decimal number."},
#     # {"role": "user", "content": "37 + 56 = "},
#     # {"role": "user", "content": "37 * 56 = "},
# ]

terminators = [
    pipe.tokenizer.eos_token_id,
    pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]
pipe.tokenizer.pad_token_id = pipe.tokenizer.eos_token_id
pipe.tokenizer.padding_side = "left"

def formatting_func(example):
    query = eval(example['query'])
    text = {"role": "user", "content": f"{example['story']}\n{query[1]} is the WHAT of {query[0]}?"}
    return text

dataset = load_dataset("CLUTRR/v1", "rob_train_clean_23_test_all_23", split="test")
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

print(pipe.__class__)
print(pipe.tokenizer.__class__)
print(pipe.model.model.__class__)

acc = 0
wrong = []
right = []
for batch in tqdm(dataloader):
    messages = []
    batch = [{key: value[i] for key, value in batch.items()} for i in range(batch_size)]
    target = []
    for x in batch:
        messages.append([prompt, formatting_func(x)])
        target.append(x['target_text'])
    outputs = forward(pipe, messages, 4)
    result = outputs
    print(result)
    answer = [y.split(' ')[-1][:-1] for y in result]
    print(answer, target)
    for i, x in enumerate(answer):
        if x.lower() == target[i].lower():
            print('correct')
            acc += 1
acc /= len(dataset)
print(f"Accuracy: {acc}")