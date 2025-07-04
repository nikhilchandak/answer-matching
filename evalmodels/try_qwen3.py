from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

class QwenChatbot:
    def __init__(self, model_path="/fast/nchandak/models/Qwen3-4B"):
        self.model = LLM(model=model_path)
        self.history = []
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
    def generate_response(self, user_input):
        # Check for thinking mode flags
        thinking_mode = True
        if "/no_think" in user_input:
            thinking_mode = False
            user_input = user_input.replace("/no_think", "").strip()
        elif "/think" in user_input:
            thinking_mode = True
            user_input = user_input.replace("/think", "").strip()
            
        user_input += "\n Think step by step and give your answer in <answer> </answer> tags."
        messages = self.history + [{"role": "user", "content": user_input}]
        messages = [{"role": "user", "content": user_input}]
        
        # Create chat template manually
        # prompt = ""
        # for msg in messages:
        #     if msg["role"] == "user":
        #         prompt += f"<|im_start|>user\n{msg['content']}<|im_end|>\n"
        #     elif msg["role"] == "assistant":
        #         prompt += f"<|im_start|>assistant\n{msg['content']}<|im_end|>\n"
        
        # prompt += "<|im_start|>assistant\n"
        
        
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=thinking_mode,
        )

        
        # Set sampling parameters
        sampling_params = SamplingParams(
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            max_tokens=1024,
            stop_token_ids=[151645],  # <|im_end|> token ID
            stop=["<|im_end|>"]
        )
        
        # Generate response
        outputs = self.model.generate(prompt, sampling_params)
        response = outputs[0].outputs[0].text.strip()
        
        # Update history
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": response})
        
        return response

# Example Usage
if __name__ == "__main__":
    chatbot = QwenChatbot()
    
    # First input (without /think or /no_think tags, thinking mode is enabled by default)
    user_input_1 = "How many r's in strawberries? /no_think"
    print(f"User: {user_input_1}")
    response_1 = chatbot.generate_response(user_input_1)
    print(f"Bot: {response_1}")
    print("----------------------")
    
    # Second input with /no_think
    user_input_2 = "Then, how many r's in blueberries? /no_think"
    print(f"User: {user_input_2}")
    response_2 = chatbot.generate_response(user_input_2)
    print(f"Bot: {response_2}")
    print("----------------------")
    
    # Third input with /think
    user_input_3 = "Are you sure? /no_think"
    print(f"User: {user_input_3}")
    response_3 = chatbot.generate_response(user_input_3)
    print(f"Bot: {response_3}")
