from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

model_name = "tiiuae/falcon-40b"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    load_in_8bit=True  # Optional for memory-efficient loading
)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
response = pipe("Explain the impact of inflation on the stock market.", max_length=200)

print(response[0]["generated_text"])
