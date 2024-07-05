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

pipe : TextGenerationPipeline = None

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

def forward_fn(model_inputs, **generate_kwargs):
    input_ids = model_inputs["input_ids"]
    # Allow empty prompts
    if input_ids.shape[1] == 0:
        input_ids = None
        attention_mask = None
        in_b = 1
    else:
        in_b = input_ids.shape[0]
    prompt_text = model_inputs.pop("prompt_text")
    attention_mask = model_inputs.get("attention_mask", None)
    generated_sequence = generate(input_ids=input_ids, attention_mask=attention_mask, **generate_kwargs)
    out_b = generated_sequence.shape[0]
    generated_sequence = generated_sequence.reshape(in_b, out_b // in_b, *generated_sequence.shape[1:])
    # TODO: compute scores and return logis, scores
    return {"generated_sequence": generated_sequence, "input_ids": input_ids, "prompt_text": prompt_text}

def generate(input_ids, attention_mask, **generate_kwargs):
    global pipe
    pad_token_id = pipe.tokenizer.pad_token_id
    eos_token_id = pipe.tokenizer.eos_token_id
    answer = ""
    unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device)
    this_peer_finished = False
    while True:
        outputs = pipe.model.forward(input_ids)
        next_tokens = torch.argmax(outputs['logits'][:, -1, :], dim=-1)
        next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
        print(eos_token_id, next_tokens)
        answer += pipe.tokenizer.decode(next_tokens, skip_special_tokens=True)
        input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], dim=1)
        unfinished_sequences = unfinished_sequences.mul(
                    next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
                )
        if unfinished_sequences.max() == 0:
                    this_peer_finished = True
        if this_peer_finished:
            break
    return input_ids

def forward(_pipe, text_inputs, num_workers):
    global pipe
    pipe = _pipe
    print(text_inputs)
    chats = [Chat(chat) for chat in text_inputs]
    batch_size = len(chats)
    preprocess_params = {}
    forward_params = {}
    postprocess_params = {}
    final_iterator = get_iterator(
        pipe, chats, num_workers, batch_size, preprocess_params, forward_fn, forward_params, postprocess_params
    )
    outputs = list(final_iterator)
    outputs = [x[0]['generated_text'][-1]['content'] for x in outputs]
    return outputs

def postprocess(outputs, **kwargs):
    #TODO: return logits and scores, maybe hidden_states
    print(outputs)
    return outputs