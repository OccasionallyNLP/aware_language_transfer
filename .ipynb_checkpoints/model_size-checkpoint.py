from transformers import AutoModelForCausalLM, AutoConfig
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    args = parser.parse_args()
    return args

def get_model_size(model, wo_lm_head=False):
    cnt = 0
    for name, i in model.named_parameters():
        if wo_lm_head and "lm_head" in name:
            continue
        
        if i.requires_grad:
            cnt += i.numel()
    return cnt

def main():
    args = get_args()
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    model = AutoModelForCausalLM.from_config(config)
    count = get_model_size(model)
    print(f"Model size: {format(count, ',')}), parameters")
    count = get_model_size(model, wo_lm_head=True)
    print(f"Model size (wo lm head): {format(count, ',')}), parameters")

if __name__ == "__main__":
    main()
