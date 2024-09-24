import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from litellm import acompletion
import asyncio
import json
import os
from pathlib import Path
from datetime import datetime

# 1. Model Setup
model_name = "microsoft/Phi-3-mini-128k-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, output_hidden_states=True)

# 2. Data Preparation
def prepare_data():
    # This is a simplified dataset. In practice, you'd have a much larger and diverse dataset.
    data = [
        {"question": "What is the capital of France?", "correct_answer": "Paris"},
        {"question": "Who wrote 'Romeo and Juliet'?", "correct_answer": "William Shakespeare"},
        # Add more questions and correct answers...
    ]
    return data

# 3. Feature Extraction and Model Answer Generation
def extract_features_and_answer(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=200, output_hidden_states=True, return_dict_in_generate=True)
    
    # Extract the last 5 layers of the last token from the input sequence
    last_5_layers = torch.cat([outputs.hidden_states[0][-i][:, -1, :] for i in range(1, 6)], dim=-1)
    
    # Decode the generated answer
    model_answer = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)

    print(model_answer)
    
    return last_5_layers.squeeze(0), model_answer

# 4. GPT-4 Mini for Labeling
async def label_with_gpt4mini(question, correct_answer, model_answer, api_key):
    prompt = f"""
    Question: {question}
    Correct Answer: {correct_answer}
    Model Answer: {model_answer}

    Is the model's answer correct? The wording does not need to be exactly the same but it should be factually correct. Please respond with either 'correct' or 'incorrect'.
    """
    messages = [{"role": "user", "content": prompt}]
    
    response = await acompletion(
        model="gpt-4o-mini",
        messages=messages,
        api_key=api_key
    )

    print(prompt)
    print(response.choices[0].message['content'])
    
    label = 1 if response.choices[0].message['content'].lower().strip() == 'correct' else 0
    return label

# 5. Save data in batches
def save_data_batch(X, y, batch_num, save_dir):
    batch_data = {
        'X': X.tolist(),
        'y': y.tolist()
    }
    with open(os.path.join(save_dir, f'batch_{batch_num}.json'), 'w') as f:
        json.dump(batch_data, f)

# 6. Generate Model Answers and Labels
async def generate_dataset(data, api_key, save_dir):
    X = []
    y = []
    batch_num = 0
    for i, item in enumerate(data):
        question = item['question']
        correct_answer = item['correct_answer']
        
        features, model_answer = extract_features_and_answer(question)
        label = await label_with_gpt4mini(question, correct_answer, model_answer, api_key)
        
        X.append(features.tolist())
        y.append(label)
        
        if (i + 1) % 10 == 0 or i == len(data) - 1:
            save_data_batch(torch.tensor(X), torch.tensor(y), batch_num, save_dir)
            batch_num += 1
            X = []
            y = []
    
    return batch_num

# 7. Load data batches
def load_data_batches(save_dir):
    X = []
    y = []
    for file in os.listdir(save_dir):
        if file.endswith('.json'):
            with open(os.path.join(save_dir, file), 'r') as f:
                batch_data = json.load(f)
                X.extend(batch_data['X'])
                y.extend(batch_data['y'])
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# 8. Logistic Regression Model
class LogisticRegressionProbe(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        return torch.sigmoid(self.linear(x)).squeeze(-1)

# 9. Training Function
def train_probe(X, y, model, optimizer, criterion, epochs=100):
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

# 10. Main execution
async def main():
    api_key = ""#Replace with API Key
    
    # Create a timestamped directory to store the batches
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path('data_batches') / timestamp
    save_dir.mkdir(parents=True, exist_ok=True)
    
    data = prepare_data()
    total_batches = await generate_dataset(data, api_key, save_dir)
    
    print(f"Total batches saved: {total_batches}")

    # Load the data
    X, y = load_data_batches(save_dir)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    input_dim = X.shape[1]
    probe = LogisticRegressionProbe(input_dim)
    optimizer = torch.optim.Adam(probe.parameters())
    criterion = nn.BCELoss()

    train_probe(X_train, y_train, probe, optimizer, criterion)

    # Evaluation
    with torch.no_grad():
        y_pred = (probe(X_test) > 0.5).float()
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy}")

    # Function to use the trained probe
    def predict_truthfulness(prompt):
        features, _ = extract_features_and_answer(prompt)
        with torch.no_grad():
            prediction = probe(features.unsqueeze(0))
        return "Likely true" if prediction.item() > 0.5 else "Likely false"

    # Example usage
    # test_prompt = "The Earth is flat."
    # print(f"Prompt: {test_prompt}")
    # print(f"Prediction: {predict_truthfulness(test_prompt)}")

if __name__ == "__main__":
    asyncio.run(main())