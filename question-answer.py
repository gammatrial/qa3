import torch
from simpletransformers.question_answering import QuestionAnsweringModel
from transformers import BertTokenizer

# Load the saved model and tokenizer
saved_model_dir = "outputs/bert/trained_model"  # Adjust based on your saved model directory
model = QuestionAnsweringModel("bert", saved_model_dir)

# Create a tokenizer object
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

# Define the context and question for which you want to generate predictions
context = "Architecturally, the school has a Catholic character. Atop the Main Building's gold dome is a golden statue of the Virgin Mary."
question = "What is on top of the Main Building?"

# Tokenize the input
inputs = tokenizer(question, context, return_tensors="pt")

# Generate predictions
outputs = model.model(**inputs)

# Decode the predicted answer
answer_start = torch.argmax(outputs.start_logits)
answer_end = torch.argmax(outputs.end_logits) + 1
answer = tokenizer.decode(inputs["input_ids"][0][answer_start:answer_end])

# Print the predicted answer
print("Predicted Answer:", answer)