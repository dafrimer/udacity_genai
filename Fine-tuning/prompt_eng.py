import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch import nn

# Load pre-trained model and tokenizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Set the model to evaluation mode
model.eval()

# Define a learned soft prompt (initialized with zeros)
soft_prompt_length = 5  # Length of the soft prompt
soft_prompt = nn.Parameter(torch.zeros(1, soft_prompt_length, model.config.n_embd))  # Learnable parameters

# Example input text related to weather
input_texts = [
    "The weather today is",
    "Tomorrow's forecast predicts",
    "In the evening, it will be"
]

# Convert input texts to tokens and concatenate with soft prompt
for input_text in input_texts:
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    # Concatenate soft prompt with input tokens
    # We need to get the embeddings for the input_ids
    input_embeddings = model.transformer.wte(input_ids)  # Get the token embeddings
    input_with_prompt = torch.cat((soft_prompt.detach(), input_embeddings), dim=1)

    # Generate predictions
    with torch.no_grad():
        outputs = model(inputs_embeds=input_with_prompt)
        predictions = outputs.logits[:, -1, :]  # Get the last token's predictions

    # Convert predictions to text
    predicted_index = torch.argmax(predictions, dim=-1)
    predicted_token = tokenizer.decode(predicted_index)

    print(f"Input: {input_text}")
    print(f"Predicted next token: {predicted_token}")
# Note: In practice, you would fine-tune the soft prompt using a dataset.