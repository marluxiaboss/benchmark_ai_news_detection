from generation import GenLoader
from utils.gen_utils import transform_chat_template_with_prompt

import sys
sys.path.append('text_quality_evaluation/human-eval')
from evalplus.data import get_human_eval_plus, write_jsonl


def generate_one_completion(prompt, gen_model, gen_config, tokenizer):
    
    #prefixes_with_prompt = [transform_chat_template_with_prompt(
    #prefix, user_prompt, gen_tokenizer,
    #use_chat_template, template_type, system_prompt, forced_prefix=prefix) for prefix in prefixes]
    
    user_prompt=""
    prompt_chat = transform_chat_template_with_prompt(
        prompt, user_prompt, tokenizer,
        use_chat_template=gen_config.use_chat_template, template_type=gen_config.chat_template_type,
        forced_prefix="")
    
    watermarking_scheme = None
    
    gen_text = gen_model([prompt_chat], batch_size=1, watermarking_scheme=watermarking_scheme)

        

    #prefixes = [prefix.strip() for prefix in prefixes]
    #fake_articles = [fake_article.strip() for fake_article in fake_articles]
    #fake_articles = [f"{prefixes[i]} {fake_articles[i]}" for i in range(len(fake_articles))]
    
    #print(f"generated text: ", {gen_text[0]})
    return gen_text[0]

def generate_batch_completions(prompts, gen_model, gen_config, tokenizer, batch_size=2):

    

    user_prompt=""
    prompts_chat = [transform_chat_template_with_prompt(
        prompt, user_prompt, tokenizer,
        use_chat_template=gen_config.use_chat_template, template_type=gen_config.chat_template_type,
        forced_prefix="") for prompt in prompts]
    
    # use strip! very important
    prompt_chats = [prompt.strip() for prompt in prompts_chat]
    
    
    print("prompts chat: ", prompts_chat)
    
    watermarking_scheme = None
    gen_texts = gen_model(prompt_chats, batch_size=batch_size, watermarking_scheme=watermarking_scheme)
    
    return gen_texts

def run_human_eval():
    
    model_name = "qwen2_0_5B"
    gen_params = {
        "max_new_tokens": 512,
        #"min_new_tokens": 200,
        "temperature": 0.2,
        "top_p": 0.95,
        "repetition_penalty": 1,
        "do_sample": True,
        "top_k": 50
    }
    
    device = "cuda"
    batch_size=2
    
    
    # load model and tokenizer
    gen_loader = GenLoader(model_name, gen_params, device)
    gen, gen_model, gen_config = gen_loader.load()
    tokenizer = gen_config.tokenizer

    problems = get_human_eval_plus()

    num_samples_per_task = 1
    #samples = [
    #    dict(task_id=task_id, completion=generate_one_completion(problems[task_id]["prompt"], gen_model, gen_config, tokenizer))
    #    for task_id in problems
    #    for _ in range(num_samples_per_task)
    #]
    
    # batch version assuming num_samples_per_task = 1
    samples = []
    prompts = [problems[task_id]["prompt"] for task_id in problems]
    task_ids = [task_id for task_id in problems]
    
    # take first n problems
    #nb_problems = len(problems)
    nb_problems = 10
    prompts = prompts[:nb_problems]
    
    
    completions = generate_batch_completions(prompts, gen_model, gen_config, tokenizer, batch_size=batch_size)
    
    # cleaning steps
    predictions= []

    for idx, prediction in enumerate(completions):
        task_id = task_ids[idx]
        code = prediction
        ref = problems[task_id]
        prompt = ref["prompt"]
        # 1. remove the prefixed prompt
        #code = code[len(prompt):]
        # 2. remove everything after "\n\n"
        code = code.split("\n\n\n")[0]
        code = code.split("\n\n")[0]
        # 3. remove everything after the "def "
        code = code.split("def ")[0]
        prediction = code
        predictions.append(prediction)
    
    for i, task_id in enumerate(problems):
        if i >= nb_problems:
            break
        samples.append(dict(task_id=task_id, completion=predictions[i]))

    write_jsonl("samples.jsonl", samples)
    
run_human_eval()