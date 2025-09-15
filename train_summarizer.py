from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer
from datasets import load_dataset

# Load dataset
dataset = load_dataset("json", data_files="dataset.json")


#Tokenisation
model_name = "facebook/bart-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def preprocess_function(examples):
    inputs = [doc for doc in examples["chat_log"]]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)
    
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["summary"], max_length=128, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Tokenize entire dataset
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Train Test split
train_dataset = tokenized_dataset["train"].shuffle(seed=42)
test_dataset = tokenized_dataset["train"].shuffle(seed=42).select(range(1))


training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

print("🚀 Starting fine-tuning...")
trainer.train()
print("✅ Fine-tuning complete.")

# ============================================================
# Save Model
# ============================================================

trainer.save_model("./fine-tuned-moderator")
tokenizer.save_pretrained("./fine-tuned-moderator")
print("💾 Model and tokenizer saved to ./fine-tuned-moderator")
