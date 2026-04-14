from transformers import GenerationConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
from tqdm import tqdm
import argparse
from peft import PeftModel
import json

def parse_args():
    parser = argparse.ArgumentParser(description="Test Llama model")

    parser.add_argument('--model_path', type=str, default='output/input/Llama-3.2-1B-Instruct/', help='model path')
    parser.add_argument('--device', type=str, default='cuda', help='device') 
    parser.add_argument('--base_model', type=str, default='meta-llama/Llama-3.2-1B-Instruct', help='base model')
    parser.add_argument('--save_path', type=str, default='output/merged_model', help='path to save the merged model')
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()
    device = torch.device(args.device)
    print(device)

    base_model = args.base_model
    model_path = args.model_path

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    local_rank = int(os.environ.get("LOCAL_RANK") or 0)

    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
            base_model,
            #torch_dtype=torch_dtype,
            device_map=device
        )
    model = PeftModel.from_pretrained(
            model,
            model_path,
            #torch_dtype=torch_dtype,
            device_map=device
        )

    model = model.merge_and_unload()

    model.save_pretrained(args.save_path)
    tokenizer.save_pretrained(args.save_path)
                

          

        
