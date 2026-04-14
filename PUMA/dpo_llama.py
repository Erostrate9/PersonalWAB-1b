import os
from torch.utils.data import Dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer, T5Config
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers import Trainer
from transformers import (
    BitsAndBytesConfig,
    HfArgumentParser,
    pipeline,
    logging,
)
import torch
import logging
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard.writer import SummaryWriter
import random
import wandb
# import deepspeed
from typing import Dict, List
from datasets import load_dataset
from peft import TaskType, LoraConfig, get_peft_model, PeftModel
import argparse
import sys
import inspect
import json
from utils import LlaMaTrainerwithTemperature
from datetime import datetime
from trl import DPOConfig, DPOTrainer
from datasets import Dataset as HFDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune Llama")

    parser.add_argument('--data_path', type=str, default='data', help='data path')
    parser.add_argument('--output_dir', type=str, default='output', help='output directory')
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-3.2-1B-Instruct', help='model name')
    parser.add_argument('--model_path', type=str, default='output/', help='model path')
    parser.add_argument('--train_epoch', type=int, default=100, help='number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--train_batch_size', type=int, default=128, help='training batch size')
    parser.add_argument('--wandb_log_freq', type=int, default=5, help='wandb log frequency')
    parser.add_argument('--source_length', type=int, default=128, help='source length')
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help='warmup ratio')
    parser.add_argument('--eval_strategy', type=str, default='epoch', help='evaluation strategy')
    parser.add_argument('--save_strategy', type=str, default='epoch', help='save strategy')
    parser.add_argument('--save_total_limit', type=int, default=5, help='save total limit')
    parser.add_argument('--logging_steps', type=int, default=100, help='logging steps')
    parser.add_argument('--deepseed_config', type=str, default=None, help='deepspeed config file')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='gradient accumulation steps')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank')
    parser.add_argument('--temperature', type=float, default=1.0, help='softmax temperature')
    parser.add_argument('--float16', action='store_true', help='use float16')
    parser.add_argument('--bf16', action='store_true', help='use bf16')
    parser.add_argument('--train_on', type=str, default='param', choices=['param'], help='train on parameter DPO pairs')
    parser.add_argument('--prompt_length', type=int, default=768, help='memory token length')
    parser.add_argument('--beta', type=float, default=0.1, help='beta')
    parser.add_argument('--min_reward_margin', type=float, default=0.0, help='Optional minimum offline reward margin filter')
    
    return parser.parse_args()


def filter_dpo_records(records, tokenizer, prompt_length, min_reward_margin):
    filtered = []
    skipped_long = 0
    skipped_margin = 0
    for item in records:
        if item.get('reward_margin') is not None and item['reward_margin'] < min_reward_margin:
            skipped_margin += 1
            continue
        prompt_ids = tokenizer(item['prompt'], truncation=False, return_tensors=None)['input_ids']
        if len(prompt_ids) > prompt_length:
            skipped_long += 1
            continue
        filtered.append(item)
    return filtered, skipped_long, skipped_margin


def build_dpo_config_kwargs(train_args, output_dir_name, reporter, learning_rate, train_batch_size, source_length, prompt_length):
    kwargs = {
        "output_dir": output_dir_name,
        "remove_unused_columns": False,
        "num_train_epochs": train_args.train_epoch,
        "per_device_train_batch_size": train_batch_size,
        "per_device_eval_batch_size": train_batch_size,
        "dataloader_num_workers": 10,
        "warmup_ratio": train_args.warmup_ratio,
        "learning_rate": learning_rate,
        "logging_dir": output_dir_name + "/logs/",
        "report_to": reporter,
        "save_strategy": train_args.save_strategy,
        "save_total_limit": train_args.save_total_limit,
        "logging_steps": train_args.logging_steps,
        "deepspeed": train_args.deepseed_config,
        "gradient_accumulation_steps": train_args.gradient_accumulation_steps,
        "fp16": train_args.float16,
        "bf16": train_args.bf16,
        "save_only_model": True,
    }
    signature = inspect.signature(DPOConfig.__init__)
    if "max_length" in signature.parameters:
        kwargs["max_length"] = source_length
    if "max_prompt_length" in signature.parameters:
        kwargs["max_prompt_length"] = prompt_length
    if "beta" in signature.parameters:
        kwargs["beta"] = train_args.beta
    if "evaluation_strategy" in signature.parameters:
        kwargs["evaluation_strategy"] = train_args.eval_strategy
    else:
        kwargs["eval_strategy"] = train_args.eval_strategy
    return kwargs


def build_dpo_trainer_kwargs(tokenizer):
    signature = inspect.signature(DPOTrainer.__init__)
    if "tokenizer" in signature.parameters:
        return {"tokenizer": tokenizer}
    return {"processing_class": tokenizer}


def load_adapter_metadata(model_path):
    if not model_path:
        return None
    adapter_config_path = os.path.join(model_path, "adapter_config.json")
    if not os.path.exists(adapter_config_path):
        return None
    return json.load(open(adapter_config_path))


if __name__ == '__main__':

    train_args = parse_args()
    data_path = train_args.data_path

    print('training on: ', data_path)

    adapter_metadata = load_adapter_metadata(train_args.model_path)
    adapter_path = train_args.model_path if adapter_metadata is not None else None
    model_name = (
        adapter_metadata.get("base_model_name_or_path")
        if adapter_metadata and adapter_metadata.get("base_model_name_or_path")
        else train_args.model_name
    )
    
    train_epoch = train_args.train_epoch
    learning_rate = train_args.learning_rate
    train_batch_size = train_args.train_batch_size
    source_length = train_args.source_length
    prompt_length = train_args.prompt_length
    current_time = datetime.now().strftime("%Y%m%d_%H%M")
    output_dir = current_time+'_'+str(data_path.split('/')[-1])+'_ep'+str(train_epoch)+'_lr'+str(learning_rate)+'_bch'+str(train_batch_size)

    output_dir_name = train_args.output_dir + '/' + model_name.split('/')[-1] + '/' + output_dir
    
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    local_rank = int(os.environ.get("LOCAL_RANK") or 0)
    if ddp:
        device_map = {"": local_rank}
    
    tokenizer_source = adapter_path if adapter_path is not None else model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if train_args.float16:
        torch_dtype = torch.float16
    elif train_args.bf16:
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32
    
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                 torch_dtype=torch_dtype,
                                                 config=config,
                                                 device_map=device_map)
    if adapter_path is not None:
        model = PeftModel.from_pretrained(
            model,
            adapter_path,
            is_trainable=True,
            torch_dtype=torch_dtype,
        )


    model.config.use_cache = False

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        modules_to_save=['embed_tokens', 'lm_head',
                         'input_layernorm', 'post_attention_layernorm', 'norm'],
        target_modules=['q_proj', 'v_proj', 'k_proj',
                        'o_proj', 'gate_proj', 'down_proj', 'up_proj'],
        lora_dropout=0,
        bias="none",
        inference_mode=False,
        task_type=TaskType.CAUSAL_LM,
    )

    reporter =  ['wandb'] if local_rank == 0 else "none"
    reporter =  'none'

    training_args = DPOConfig(
        **build_dpo_config_kwargs(
            train_args=train_args,
            output_dir_name=output_dir_name,
            reporter=reporter,
            learning_rate=learning_rate,
            train_batch_size=train_batch_size,
            source_length=source_length,
            prompt_length=prompt_length,
        )
    )
 
    raw_train_dataset = list(load_dataset('json', data_files=data_path, field='train')['train'])
    raw_test_dataset = list(load_dataset('json', data_files=data_path, field='test')['train'])
    train_records, train_skipped_long, train_skipped_margin = filter_dpo_records(
        raw_train_dataset,
        tokenizer=tokenizer,
        prompt_length=prompt_length,
        min_reward_margin=train_args.min_reward_margin,
    )
    test_records, test_skipped_long, test_skipped_margin = filter_dpo_records(
        raw_test_dataset,
        tokenizer=tokenizer,
        prompt_length=prompt_length,
        min_reward_margin=train_args.min_reward_margin,
    )
    train_dataset = HFDataset.from_list(train_records)
    test_dataset = HFDataset.from_list(test_records)

    print('train dataset size: ', len(train_dataset))
    print('test dataset size: ', len(test_dataset))
    print('train skipped overlong prompts: ', train_skipped_long)
    print('test skipped overlong prompts: ', test_skipped_long)
    print('train skipped low-margin pairs: ', train_skipped_margin)
    print('test skipped low-margin pairs: ', test_skipped_margin)

    if local_rank == 0:
        os.makedirs(output_dir_name, exist_ok=True)
        logging.basicConfig(filename=output_dir_name+'/training_log.log', level=logging.INFO, format='%(asctime)s - %(message)s')
        logger = logging.getLogger(__name__)

        logger.info('traing arguments: '+str(train_args))
        logger.info('training dataset size: '+str(len(train_dataset)))
        logger.info('test dataset size: '+str(len(test_dataset)))
        logger.info('train skipped overlong prompts: '+str(train_skipped_long))
        logger.info('test skipped overlong prompts: '+str(test_skipped_long))
        logger.info('train skipped low-margin pairs: '+str(train_skipped_margin))
        logger.info('test skipped low-margin pairs: '+str(test_skipped_margin))
        logger.info('transfomers training_args: '+str(training_args))

    trainer = DPOTrainer(
        model=model,
        #ref_model=model_ref,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        #data_collator=data_collator,
        peft_config=None if adapter_path is not None else lora_config,
        **build_dpo_trainer_kwargs(tokenizer),
    )

    trainer.train()
