from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import joblib

# # Load the saved model
# log_reg = joblib.load('/Users/dhruvroongta/Downloads/logistic_regression_model.joblib')

# # Use the loaded model for prediction
# y_pred = log_reg.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Accuracy: {accuracy:.4f}")


# Load the falcon-7b model and tokenizer from Hugging Face
model_name = "tiiuae/falcon-7b"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Ensure the model is in evaluation mode
model.eval()

# Define a function to generate text


def generate_text(prompt, max_length=50):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"], max_length=max_length, do_sample=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# Example usage
prompt = "Once upon a time"
generated_text = generate_text(prompt)
print(f"Generated text: {generated_text}")
