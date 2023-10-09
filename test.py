from langchain import HuggingFacePipeline
from transformers import AutoTokenizer, pipeline
import torch
import json
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

model = "tiiuae/falcon-7b-instruct" #tiiuae/falcon-40b-instruct
#model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-40b", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model)

# Create the pipeline using the loaded tokenizer and model
pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
    max_length=500,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id
)

with open('xyz.json') as f:
	data = json.load(f)

llm = HuggingFacePipeline(pipeline = pipeline, model_kwargs = {'temperature':0})

from langchain import PromptTemplate,  LLMChain

# template = "Write the code to make a form in HTML in which the fields are given as follows\n"
# for key, value in data.items():
#     if value:
#         template += " Field Name: "+str(key)+"\n"

template = ''' Add the address functionality in this person class :
class Person "{
    private String name;
    private int age;
    private String gender;

    public Person(String name, int age, String gender) {
        this.name = name;
        this.age = age;
        this.gender = gender;
    }

    public String getName() {
        return name;
    }

    public int getAge() {
        return age;
    }

    public String getGender() {
        return gender;
    }
}
'''
    

prompt = PromptTemplate(template=template, input_variables=[])

llm_chain = LLMChain(prompt=prompt, llm=llm)

print(llm_chain.run({}))
