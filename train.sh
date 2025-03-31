accelerate launch train.py --config_name ./data/micro_llama --tokenizer_name ./data/micro_llama --dataset_config_path configs/token_based.yml --output_dir output --max_train_steps 3000 --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --learning_rate 3e-4 --weight_decay 0.01 --num_warmup_steps 100 --gradient_accumulation_steps 1 --evaluation_steps 5 --block_size 1024 --lr_scheduler_type cosine_with_min_lr --min_learning_rate 3e-5 --checkpointing_steps 1
