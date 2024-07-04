from transformers import pipeline
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers.pipelines.text_generation import TextGenerationPipeline, Chat
from transformers.models.llama import LlamaForCausalLM
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from tqdm import tqdm
from myPipeline import MyPipeline
from transformers.pipelines.pt_utils import PipelineDataset, PipelineIterator
from transformers.pipelines.base import pad_collate_fn, no_collate_fn, Pipeline

def get_iterator(
    self: Pipeline, inputs, num_workers: int, batch_size: int, preprocess_params, forward_fn, forward_params, postprocess_params
):
    dataset = PipelineDataset(inputs, self.preprocess, preprocess_params)
    # TODO hack by collating feature_extractor and image_processor
    feature_extractor = self.feature_extractor if self.feature_extractor is not None else self.image_processor
    collate_fn = no_collate_fn if batch_size == 1 else pad_collate_fn(self.tokenizer, feature_extractor)
    dataloader = DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, collate_fn=collate_fn)
    model_iterator = PipelineIterator(dataloader, forward_fn, forward_params, loader_batch_size=batch_size)
    final_iterator = PipelineIterator(model_iterator, self.postprocess, postprocess_params)
    return final_iterator

# def forward_fn(model_inputs, **forward_params):
    # generated_sequence = generate(input_ids=input_ids, attention_mask=attention_mask, **generate_kwargs)
    # out_b = generated_sequence.shape[0]
    # if self.framework == "pt":
    #     generated_sequence = generated_sequence.reshape(in_b, out_b // in_b, *generated_sequence.shape[1:])
    # elif self.framework == "tf":
    #     generated_sequence = tf.reshape(generated_sequence, (in_b, out_b // in_b, *generated_sequence.shape[1:]))
    # return {"generated_sequence": generated_sequence, "input_ids": input_ids, "prompt_text": prompt_text}

def generate(pipe, messages):
    chat_prompt = Chat(messages)
    input_prompt = pipe.tokenizer.apply_chat_template(
                    chat_prompt.messages,
                    truncation=None,
                    padding=False,
                    max_length=None,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                    tokenize=False
                )
    input_tokenized = pipe.tokenizer.tokenize(input_prompt)
    # new_tokenized = []
    # for t in input_tokenized:
    #     if t.isdigit() and len(t) > 1:
    #         new_tokenized.extend(list(t))
    #     else:
    #         new_tokenized.append(t)
    # input_tokenized = new_tokenized
    input_ids = pipe.tokenizer.convert_tokens_to_ids(input_tokenized)
    input_ids = torch.tensor(input_ids).unsqueeze(0)
    answer = ""
    while True:
        outputs = pipe.model.forward(input_ids)
        next_token_id = torch.argmax(outputs['logits'][:, -1, :])
        answer += pipe.tokenizer.decode(next_token_id, skip_special_tokens=True)
        input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0).unsqueeze(0)], dim=1)
        if next_token_id == pipe.tokenizer.eos_token_id:
            break
    return answer

# terminators = [
#     pipe.tokenizer.eos_token_id,
#     pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")
# ]
# pipe.tokenizer.pad_token_id = pipe.tokenizer.eos_token_id
# pipe.tokenizer.padding_side = "left"

# outputs = pipe(
#     messages,
#     max_new_tokens=10,
#     eos_token_id=terminators,
#     early_stopping=True,
#     output_logits=True,
#     return_dict_in_generate=True,
# )

# print(outputs)