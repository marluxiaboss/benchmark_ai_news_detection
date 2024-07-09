#!/bin/bash

#python create_dataset.py --dataset_size=$dataset_size --max_sample_len=$max_sample_len --prefix_size=$prefix_size --dataset_name=$dataset_name --generator_name=$generator_name --attack_name=$attack_name --watermarking_scheme_name=$watermarking_scheme_name --batch_size=$batch_size --device=$device
#python create_dataset.py generation=gen_params_attack generation.temperature=1.0
#python create_dataset.py generation=prompt_paraphrasing_attack watermark=watermark_kgw generation.dataset_size=10 experiment_name=fix_bug
#python test_detector.py detection=fast_detect_gpt generation=prompt_paraphrasing_attack watermark=watermark_kgw generation.experiment_name=fix_bug
python test_detector.py detection=watermark_detector generation=prompt_paraphrasing_attack watermark=watermark_kgw generation.experiment_name=fix_bug

