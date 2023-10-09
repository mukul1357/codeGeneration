import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training
from peft import PeftConfig,PeftModel
from trl import SFTTrainer
from peft import get_peft_model
from peft import LoraConfig, TaskType
from datasets import load_dataset


dataset = load_dataset("sahil2801/CodeAlpaca-20k", split='train')

def format_input(example: dict)->str:
  example['formatted_text'] = f''' ### User: Please write the code according to the following instruction and input: Instruction: {example["instruction"]} Input: {example["input"]}
  ### Code Generator: {example["output"]}'''
  return example

dataset = dataset.map(format_input)

dataset_sampled = dataset.shuffle(seed=42)




model_name = "tiiuae/falcon-7b-instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    trust_remote_code=True,
    device_map='auto'
)

model.config.use_cache = False
model.gradient_checkpointing_enable()

model = prepare_model_for_kbit_training(model)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code = True)
tokenizer.pad_token = tokenizer.eos_token



peft_config = LoraConfig(
    lora_alpha = 32,
    lora_dropout = 0.05,
    r=16,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["query_key_value"]
)


model = get_peft_model(model, peft_config)

from transformers import TrainingArguments
training_arguments = TrainingArguments(
    output_dir = './training_result',
    per_device_train_batch_size=16,
    gradient_accumulation_steps=4,
    num_train_epochs=5,
    learning_rate=2e-4,
    fp16=True,
    optim='paged_adamw_8bit',
    lr_scheduler_type="cosine",
    warmup_ratio=0.05
)



trainer=SFTTrainer(
    model=model,
    train_dataset=dataset_sampled,
    peft_config=peft_config,
    dataset_text_field="formatted_text",
    tokenizer=tokenizer,
    args=training_arguments,
    max_seq_length=1024
)

trainer.train()

trained_model_dir='./trained_model'
model.save_pretrained(trained_model_dir)

config = PeftConfig.from_pretrained(trained_model_dir)

trained_model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    return_dict=True,
    quantization_config=bnb_config,
    trust_remote_code=True,
    device_map='auto'
)

trained_model = PeftModel.from_pretrained(trained_model, trained_model_dir)
trained_model_tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path, trust_remote_code=True)
trained_model_tokenizer.pad_token = trained_model_tokenizer.eos_token

generation_config = trained_model.generation_config
generation_config.max_new_token = 1024
generation_config.temperature = 0.7
generation_config.top_p=0.7
generation_config.num_return_sequence=1
generation_config.pad_token_id=trained_model_tokenizer.pad_token_id
generation_config.eos_token_id=trained_model_tokenizer.eos_token_id

device = 'cuda:0'

instruction = '''Design a class for representing a person in Python '''
input = ''' '''

prompt = f''' ### User: Please write the code according to the following instruction and input: Instruction: {instruction} Input: {input}
  ### Code Generator: '''

encodings = trained_model_tokenizer(prompt, return_tensors='pt').to(device)

with torch.inference_mode():
  outputs=trained_model.generate(
      input_ids=encodings.input_ids,
      attention_mask=encodings.attention_mask,
      generation_config=generation_config,
      max_new_tokens=300
  )

outputs = trained_model_tokenizer.decode(outputs[0], skip_special_tokens=True)

print(outputs)
