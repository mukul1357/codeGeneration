import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftConfig,PeftModel

trained_model_dir='./trained_model'

config = PeftConfig.from_pretrained(trained_model_dir)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

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

instruction = '''Design a class of person in Java '''
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