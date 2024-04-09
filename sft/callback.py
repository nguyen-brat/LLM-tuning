from transformers.integrations import WandbCallback
from transformers import GenerationConfig
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
import torch
import wandb
from tqdm import tqdm

class LLMSampleCB(WandbCallback):
    def __init__(self, trainer, test_dataset, num_samples=10, max_new_tokens=256, log_model="checkpoint"):
	    #"A CallBack to log samples a wandb.Table during training"
        super().__init__()
        self._log_model = log_model
        self.sample_dataset = test_dataset.select(range(num_samples))
        self.model, self.tokenizer = trainer.model, trainer.tokenizer
        self.gen_config = GenerationConfig.from_pretrained(trainer.model.name_or_path,
                                                           max_new_tokens=max_new_tokens)
    def generate(self, prompt):
        tokenized_prompt = self.tokenizer(prompt, return_tensors='pt')['input_ids'].cuda()
        with torch.inference_mode():
            output = self.model.generate(tokenized_prompt, generation_config=self.gen_config)
        return self.tokenizer.decode(output[0][len(tokenized_prompt[0]):], skip_special_tokens=True)
    
    def samples_table(self, examples):
	    #"Create a wandb.Table to store the generations"
        records_table = wandb.Table(columns=["prompt", "generation"] + list(self.gen_config.to_dict().keys()))
        for example in tqdm(examples, leave=False):
            prompt = example["text"]
            generation = self.generate(prompt=prompt)
            records_table.add_data(prompt, generation, *list(self.gen_config.to_dict().values()))
        return records_table
        
    def on_evaluate(self, args, state, control,  **kwargs):
	    #"Log the wandb.Table after calling trainer.evaluate"
        super().on_evaluate(args, state, control, **kwargs)
        records_table = self.samples_table(self.sample_dataset)
        self._wandb.log({"sample_predictions":records_table})

class SaveDeepSpeedPeftModelCallback(TrainerCallback):
    def __init__(self, trainer, save_steps=500):
        self.trainer = trainer
        self.save_steps = save_steps

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if (state.global_step + 1) % self.save_steps == 0:
            self.trainer.accelerator.wait_for_everyone()
            state_dict = self.trainer.accelerator.get_state_dict(self.trainer.deepspeed)
            unwrapped_model = self.trainer.accelerator.unwrap_model(self.trainer.deepspeed)
            if self.trainer.accelerator.is_main_process:
                unwrapped_model.save_pretrained(args.output_dir, state_dict=state_dict)
            self.trainer.accelerator.wait_for_everyone()
        return control