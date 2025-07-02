from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import os

MODEL_PATH = "qwen/Qwen1.5-1.8B-Chat"
ADAPTER_PATH = os.path.join(os.getcwd(), "qwen-lora-safety-model")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(
    ADAPTER_PATH,  # Load tokenizer from fine-tuned model dir
    trust_remote_code=True,
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True
)
model = PeftModel.from_pretrained(model, ADAPTER_PATH)
model = model.to(device)
model.eval()
def get_chatbot_response(user_message, location="Unknown"):
    # Short circuit for predefined safety tips
    keywords = ["disturbance", "attack", "fire", "emergency", "report"]
    if any(word in user_message.lower() for word in keywords):
        return (
            "⚠️ Public Safety Reporting Instructions:\n\n"
            "• Report the exact location with a clear hazard category.\n"
            "• Avoid the area if possible and prioritize your safety.\n"
            "• Do not engage directly—observe from a distance.\n\n"
            "🙏 Thank you for supporting community safety."
        )

    # Use formatted prompt
    prompt = f"""
🚨 Safety Report Requested

Location: {location}
Incident: {user_message}

"""

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,               
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2,        
            num_beams=1
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if response.startswith(prompt.strip()):
        response = response[len(prompt.strip()):]

    # Format response: convert newlines to <br>
    response = response.strip().replace("\n", "<br>")

    return response

