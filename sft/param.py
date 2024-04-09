import logging
from dataclasses import dataclass, field
from typing import Optional, Union, List
from transformers import MODEL_FOR_CAUSAL_LM_MAPPING
from transformers.utils.versions import require_version

#from .param import ModelArguments, DataTrainingArguments, LoraArguments
#from .callbacks import *


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.39.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

#######################################################################################
@dataclass
class LoraArguments:
    use_peft: bool = field(
        default=True,
        metadata={
            "help": "use peft training for model"
        },
    )
    merge_adapter: bool = field(
        default=False,
        metadata={
            "help":"add adapter to model and save at end"
        }
    )
    target_modules: Optional[str] = field(
        default="all-linear",
        metadata={
            "help":"layer to lora"
        }
    )
    lora_r: int = field(
        default=16,
        metadata={
            "help": "lora r"
        }
    )
    lora_alpha: int = field(
        default=32,
        metadata={
            "help": "lora alpha"
        }
    )
    lora_dropout: int = field(
        default=0.05,
        metadata={
            "help": "lora dropout"
        }
    )
    quantization: Optional[int] = field(
        default=None,
        metadata={
            "help": "train with which qlora if None train normal without quantization. 0 mean no quantization",
            "choices": [4, 8, 0],
        }
    )
    lora_bias: str = field(
        default="none",
        metadata={
            "help": "lora bias"
        }
    )

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    model_max_length: int = field(
        default=None, metadata={"help": "max token of input and output"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    use_auth_token: bool = field(
        default=None,
        metadata={
            "help": "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead."
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option "
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
                "execute code present on the Hub on your local machine."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded. "
                "set True will benefit LLM loading time and RAM consumption."
            )
        },
    )
    use_flash_attention_2: Optional[bool] = field(
        default=False, metadata={"help":"use flash attention for training"}
    )

    merge_adapters: bool = field(
        default=False,
        metadata={
            "help": ("merge lora adapter and save in the output dir")
        },
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    response_template: str = field(
        default="###Synthetic",
        metadata={"help": "String sign where the model start to response the users"},
    )
    instruction_template: str = field(
        default=None,
        metadata={"help": "String sign where the user start instruction"},
    )
    instruction_text_column: str = field(
        default="instruction",
        metadata={"help": "String sign where the user start instruction"},
    )
    response_text_column: str = field(
        default="response",
        metadata={"help": "String sign where the user start instruction"},
    )
    train_type: str = field(
        default="intruction-sft",
        metadata={
            "help": "insutruction-sft if you want to train with instruction tuning and unsupervise-tune if you want to tune with Auto-regression on full dataset",
            "choices": ["intruction-sft", "unsupervise-tune"]
        },
    )
    patient: int = field(
        default=9999, metadata={"help": "if in a given number of epoch or step the val acc or loss not improve training process will be corrupt"}
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."