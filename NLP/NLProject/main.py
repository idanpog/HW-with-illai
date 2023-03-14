# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


from transformers import T5Tokenizer, T5ForConditionalGeneration, DataCollatorForSeq2Seq
# from transformers import T5Tokenizer, T5Model

tokenizer = T5Tokenizer.from_pretrained("t5-large")
model = T5ForConditionalGeneration.from_pretrained("t5-large")

collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

text = 'Leider ist die konventionelle Analyse zum Zusammenbruch von Lehman Brothers Wunschdenken.'
text = 'Unfortunately, the conventional post-mortem on Lehman Brothers is wishful thinking.'
input_ids = tokenizer(
    'Translate English to German:' + text, return_tensors="pt").input_ids  # Batch size 1
labels = tokenizer('Unfortunately, the conventional post-mortem on Lehman Brothers is wishful thinking.', return_tensors='pt').input_ids
# loss = model(input_ids=input_ids, labels=labels).loss
# forward pass
# loss = model(input_ids=input_ids, labels=labels)
# last_hidden_states = outputs.last_hidden_state
outputs = model.generate(input_ids)
decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(decoded)