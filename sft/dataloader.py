import numpy as np
import json
import pandas as pd
import warnings
from typing import Optional, List, Union, Dict, Any

from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)
from torch.utils.data import DataLoader, Dataset


class LazyCustomDataloader(Dataset):
    def __init__(self, tokenizer_kwargs, file_paths, max_length):
        self.tokenizer = AutoTokenizer.from_pretrained(**tokenizer_kwargs)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.load(file_paths)

        # if tune_type == 'intruction-sft':
        #     self.collator = DataCollatorForCompletionOnlyLM(response_template, instruction_template=instruction_template, tokenizer=self.tokenizer, mlm=False)
        # elif tune_type == 'unsupervise-tune':
        #     self.collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)
        self.max_length = max_length
        self.cached_data_dict = {}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if idx in self.cached_data_dict:
            return self.cached_data_dict[idx]
        message = [
            {'role': 'user', 'content':self.data["instruction"][idx]},
            {'role': 'assistant', 'content':self.data["response"][idx]}
        ]
        output_text = self.tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True
        )
        outputs_ids = self.tokenizer(
            [output_text],
            max_length=self.max_length,
            # padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=False,
        )
        #torch_input_ids = self.collator.torch_call(outputs_ids.input_ids)
        output = dict(
            input_ids=outputs_ids["input_ids"][0],
            #labels=torch_input_ids["labels"][0],
            # attention_mask=torch.Tensor(outputs_ids["attention_mask"][0]),
            attention_mask=outputs_ids["attention_mask"][0],
        )
        self.cached_data_dict[idx] = output
        return output

    def load(self, file_path):
        extension = file_path.split('.')[-1]
        if extension == 'json':
            with open(file_path, 'r') as f:
                self.data = json.load(f)
        elif extension == 'csv':
            self.data = pd.read_csv(file_path)
        else:
            ValueError("this file extension is not currently support")


class DataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):
    """
    Data collator used for completion tasks. It ensures that all the tokens of the labels are set to an 'ignore_index'
    when they do not come from the assistant. This ensure that the loss is only
    calculated on the completion made by the assistant.

    Args:
        response_template (`Union[str, List[int]]`): the template form that indicates the start of the response, typically something like
            '### Response:\n'. It can also be passed as tokenized ids, which can be useful when using a tokenizer that encodes the response
            differently if it does not have proper context.
        instruction_template (`Union[str, List[int]]`): the template form that indicates the start of the human instruction, typically something like
            '### Human:\n'. Useful for assistant-style conversation datasets. It can also be passed as tokenized ids.
        mlm (`bool`, *optional*, defaults to `False`): Whether or not to use masked language modeling in the underlying
            `DataCollatorForLanguageModeling` class. Note that this option currently has no effect but is present
             for flexibility and backwards-compatibility.
        ignore_index (`int`, *optional*, defaults to `-100`):
            The index to use to ignore the initial tokens with
    """

    def __init__(
        self,
        response_template: Union[str, List[int]],
        instruction_template: Optional[Union[str, List[int]]] = None,
        *args,
        mlm: bool = False,
        ignore_index: int = -100,
        **kwargs,
    ):
        super().__init__(*args, mlm=mlm, **kwargs)

        self.instruction_template = instruction_template
        if isinstance(instruction_template, str):
            # The user provides a string, must tokenize
            self.instruction_token_ids = self.tokenizer.encode(self.instruction_template, add_special_tokens=False)
        else:
            # The user already provides the token ids
            self.instruction_token_ids = instruction_template

        self.response_template = response_template
        if isinstance(response_template, str):
            # The user provides a string, must tokenize
            self.response_token_ids = self.tokenizer.encode(self.response_template, add_special_tokens=False)
        else:
            # The user already provides the token ids
            self.response_token_ids = response_template

        if not self.mlm and self.instruction_template and self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            warnings.warn(
                "The pad_token_id and eos_token_id values of this tokenizer are identical. "
                "If you are planning for multi-turn training, "
                "it can result in the model continuously generating questions and answers without eos token. "
                "To avoid this, set the pad_token_id to a different value."
            )

        self.ignore_index = ignore_index

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        batch = super().torch_call(examples)

        if self.instruction_template is None:
            for i in range(len(examples)):
                response_token_ids_start_idx = None

                for idx in np.where(batch["labels"][i] == self.response_token_ids[0])[0]:
                    # `response_token_ids` is `'### Response:\n'`, here we are just making sure that the token IDs match
                    if (
                        self.response_token_ids
                        == batch["labels"][i][idx : idx + len(self.response_token_ids)].tolist()
                    ):
                        response_token_ids_start_idx = idx

                if response_token_ids_start_idx is None:
                    warnings.warn(
                        f"Could not find response key `{self.response_template}` in the "
                        f'following instance: {self.tokenizer.decode(batch["input_ids"][i])} '
                        f"This instance will be ignored in loss calculation. "
                        f"Note, if this happens often, consider increasing the `max_seq_length`."
                    )
                    batch["labels"][i, :] = self.ignore_index
                else:
                    response_token_ids_end_idx = response_token_ids_start_idx + len(self.response_token_ids)

                    # Make pytorch loss function ignore all tokens up through the end of the response key
                    batch["labels"][i, :response_token_ids_end_idx] = self.ignore_index

        else:
            for i in range(len(examples)):
                response_token_ids_idxs = []
                human_token_ids_idxs = []

                for assistant_idx in np.where(batch["labels"][i] == self.response_token_ids[0])[0]:
                    # find the indexes of the start of a response.
                    if (
                        self.response_token_ids
                        == batch["labels"][i][assistant_idx : assistant_idx + len(self.response_token_ids)].tolist()
                    ):
                        response_token_ids_idxs.append(assistant_idx + len(self.response_token_ids))

                if len(response_token_ids_idxs) == 0:
                    warnings.warn(
                        f"Could not find response key `{self.response_template}` in the "
                        f'following instance: {self.tokenizer.decode(batch["input_ids"][i])} '
                        f"This instance will be ignored in loss calculation. "
                        f"Note, if this happens often, consider increasing the `max_seq_length`."
                    )
                    batch["labels"][i, :] = self.ignore_index

                human_token_ids = self.instruction_token_ids
                for human_idx in np.where(batch["labels"][i] == human_token_ids[0])[0]:
                    # find the indexes of the start of a human answer.
                    if human_token_ids == batch["labels"][i][human_idx : human_idx + len(human_token_ids)].tolist():
                        human_token_ids_idxs.append(human_idx)

                if len(human_token_ids_idxs) == 0:
                    warnings.warn(
                        f"Could not find instruction key `{self.instruction_template}` in the "
                        f'following instance: {self.tokenizer.decode(batch["input_ids"][i])} '
                        f"This instance will be ignored in loss calculation. "
                        f"Note, if this happens often, consider increasing the `max_seq_length`."
                    )
                    batch["labels"][i, :] = self.ignore_index

                if (
                    len(human_token_ids_idxs) > 0
                    and len(response_token_ids_idxs) > 0
                    and human_token_ids_idxs[0] > response_token_ids_idxs[0]
                ):
                    human_token_ids_idxs = [0] + human_token_ids_idxs

                for idx, (start, end) in enumerate(zip(human_token_ids_idxs, response_token_ids_idxs)):
                    # Make pytorch loss function ignore all non response tokens
                    if idx != 0:
                        batch["labels"][i, start:end] = self.ignore_index
                    else:
                        batch["labels"][i, :end] = self.ignore_index

                if len(response_token_ids_idxs) < len(human_token_ids_idxs):
                    batch["labels"][i, human_token_ids_idxs[-1] :] = self.ignore_index
        return batch

def get_dataloader_collator(tokenizer_kwargs, file_paths, max_length, response_template=None, instruction_template=None, tune_type='intruction-sft'):
    train_dataloader = LazyCustomDataloader(tokenizer_kwargs, file_paths["train"], max_length)
    if "validation" in file_paths.keys():
        val_dataloader = LazyCustomDataloader(tokenizer_kwargs, file_paths["validation"], max_length)
    tokenizer = AutoTokenizer.from_pretrained(**tokenizer_kwargs)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    if tune_type == 'intruction-sft':
        data_collator = DataCollatorForCompletionOnlyLM(response_template=response_template, instruction_template=instruction_template, tokenizer=tokenizer, mlm=False)
    elif tune_type == 'unsupervise-tune':
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    if "validation" in file_paths.keys():
        dataset_loader = dict(train_dataset=train_dataloader, eval_dataset=val_dataloader)
    else:
        dataset_loader = dict(train_dataset=train_dataloader)
    return dataset_loader, data_collator

if __name__ == '__main__':
    # tokenizer_kwargs = {
    #     "pretrained_model_name_or_path":"mistralai/Mistral-7B-Instruct-v0.2",
    #     "padding_side":"right"
    # }
    # data_path = {
    #     "train":"/home/nguyen/code/llm-tune/data/combine_multi_view/train/combine_pedestrian_vid_name.csv",
    #     "validation":"/home/nguyen/code/llm-tune/data/combine_multi_view/train/combine_pedestrian_vid_name.csv"
    # }
    # loader, collator = get_dataloader_collator(
    #     tokenizer_kwargs = tokenizer_kwargs,
    #     file_paths=data_path["train"],
    #     max_length=2048,
    #     response_template="###Synthetic",
    #     instruction_template="###Captions",
    # )
    # full_loader = DataLoader(loader, batch_size=4, collate_fn=collator)
    # for i in full_loader:
    #     print(i)
    #     exit()
    pass