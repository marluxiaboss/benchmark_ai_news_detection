from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
#tokenizer.eos_token ='<|eot_id|>'
tokenizer.pad_token = '<|eot_id|>'
tokenizer.padding_side = "left"

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]

input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

input_ids_non_tokenized = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=False
)

print("input_ids_non_tokenized: ", [input_ids_non_tokenized])

text_to_append = "Arrrr, me hearty! Me name be Captain Chat mate,"

input_ids_non_tokenized = f"{input_ids_non_tokenized}\n{text_to_append}"

input_ids_non_tokenized = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nContinue to write this news article: Harry Potter star Daniel Radcliffe gains access to a reported<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n\nHarry Potter star Daniel Radcliffe gains access to a reported"
#input_ids = tokenizer(input_ids_non_tokenized, return_tensors="pt").input_ids.to(model.device)

max_length = 200
#samples = ['<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nContinue to write this news article: Harry Potter star Daniel Radcliffe gains access to a reported<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n\nHarry Potter star Daniel Radcliffe gains access to a reported', '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nContinue to write this news article: The ninth floor of the Miami-Dade pretrial detention facility is<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n\nThe ninth floor of the Miami-Dade pretrial detention facility is', '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nContinue to write this news article: Drivers who were on the Minneapolis bridge when it collapsed<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n\nDrivers who were on the Minneapolis bridge when it collapsed']
#samples = ['<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nContinue to write this news article: Harry Potter star Daniel Radcliffe gains access to a reported<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nHarry Potter star Daniel Radcliffe gains access to a reported', '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nContinue to write this news article: The ninth floor of the Miami-Dade pretrial detention facility is<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n\nThe ninth floor of the Miami-Dade pretrial detention facility is', '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nContinue to write this news article: Drivers who were on the Minneapolis bridge when it collapsed<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nDrivers who were on the Minneapolis bridge when it collapsed']
samples = ['<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nContinue to write this news article: Harry Potter star Daniel Radcliffe gains access to a reported<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n\nHarry Potter star Daniel Radcliffe gains access to a reported', '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nContinue to write this news article: The ninth floor of the Miami-Dade pretrial detention facility is<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n\nThe ninth floor of the Miami-Dade pretrial detention facility is', '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nContinue to write this news article: Drivers who were on the Minneapolis bridge when it collapsed<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n\nDrivers who were on the Minneapolis bridge when it collapsed', "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nContinue to write this news article: Doctors removed five small polyps from President Bush's colon on<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n\nDoctors removed five small polyps from President Bush's colon on", '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nContinue to write this news article: The National Football League has indefinitely suspended Atlanta Falcons quarterback<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n\nThe National Football League has indefinitely suspended Atlanta Falcons quarterback', "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nContinue to write this news article: Dressed in a Superman shirt, 5-year-old Youssif held his sister's<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n\nDressed in a Superman shirt, 5-year-old Youssif held his sister's", '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nContinue to write this news article: The women are too afraid and ashamed to show their<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n\nThe women are too afraid and ashamed to show their', '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nContinue to write this news article: A key rebel commander and fugitive from a U.S. drug<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n\nA key rebel commander and fugitive from a U.S. drug', '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nContinue to write this news article: White House press secretary Tony Snow, who is undergoing treatment<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n\nWhite House press secretary Tony Snow, who is undergoing treatment', '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nContinue to write this news article: Police and FBI agents are investigating the discovery of an<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n\nPolice and FBI agents are investigating the discovery of an', '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nContinue to write this news article: As he awaits a crucial progress report on Iraq, President<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n\nAs he awaits a crucial progress report on Iraq, President', '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nContinue to write this news article: A chronology of bombings and attempted bomb attacks in the<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n\nA chronology of bombings and attempted bomb attacks in the', "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nContinue to write this news article: Carlos Alberto, who scored in FC Porto's Champions League final<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n\nCarlos Alberto, who scored in FC Porto's Champions League final", '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nContinue to write this news article: Vice President Dick Cheney will serve as acting president briefly<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n\nVice President Dick Cheney will serve as acting president briefly', '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nContinue to write this news article: A magnitude 4.2 earthquake shook the San Francisco area Friday<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n\nA magnitude 4.2 earthquake shook the San Francisco area Friday', '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nContinue to write this news article: There is "no remaining hope" of finding six men trapped<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n\nThere is "no remaining hope" of finding six men trapped']
encoding = tokenizer(
            samples, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
input_ids = encoding['input_ids'].to(model.device)

input_ids_cpu = list(input_ids.cpu())
attention_mask_cpu = list(encoding["attention_mask"].cpu())

for i, sample in enumerate(input_ids_cpu):
    print("sample: ", input_ids_cpu[i])
    print("attention_mask: ", attention_mask_cpu[i])
    
print("encoding[attention_mask]: ", encoding["attention_mask"])
#print("input_ids: ", input_ids)

outputs_ids = model.generate(
    input_ids,
    max_new_tokens=256,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
    pad_token_id=tokenizer.pad_token_id
)
#response = outputs[0][input_ids.shape[-1]:]
decoded_outputs = tokenizer.batch_decode(outputs_ids[:, input_ids.shape[1]:])

print(decoded_outputs)