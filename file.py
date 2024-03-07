import torch
import time
from transformers import LoRAForCausalLM, LoRATokenizer
from torch.cuda.amp import autocast
from transformers import optimize_model
from torch.utils.mobile_optimizer import optimize_for_mobile

def optimize_and_infer(model_path, input_text, num_runs=1):
    # Load LoRAForCausalLM model and tokenizer from Hugging Face
    model = LoRAForCausalLM.from_pretrained(model_path)
    tokenizer = LoRATokenizer.from_pretrained(model_path)

    # Tokenize user input text
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    # Optimize model using the optimize_model function from transformers library
    optimized_model = optimize_model(model, input_ids)

    # Additional optimizations (quantization and JIT compilation)
    quantized_model = torch.quantization.quantize_dynamic(
        optimized_model, {torch.nn.Linear}, dtype=torch.qint8
    )

    # Convert the model for mobile deployment (optional)
    mobile_optimized_model = optimize_for_mobile(quantized_model)

    total_tokens = 0
    total_time = 0.0

    for _ in range(num_runs):
        # Inference with warmup iterations
        num_warmup_steps = 10
        with torch.no_grad():
            # Warmup iterations
            for _ in range(num_warmup_steps):
                _ = optimized_model.generate(input_ids)

            # Measure inference time and tokens processed
            start_time = time.time()
            with autocast():
                output = optimized_model.generate(input_ids, max_length=128)
            end_time = time.time()

            total_tokens += input_ids.size(1) * num_warmup_steps + output.size(1)
            total_time += end_time - start_time

    # Average throughput and inference time
    avg_throughput = total_tokens / total_time
    avg_inference_time = total_time / num_runs

    # Decode and print the generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print("Generated Text:", generated_text)

    # Display performance metrics
    print(f"\nPerformance Metrics (Average over {num_runs} runs):")
    print(f"Average Throughput: {avg_throughput:.2f} tokens/second")
    print(f"Average Inference Time: {avg_inference_time:.4f} seconds")

if __name__ == "__main__":
    # Example usage with a Hugging Face model path and user input prompt
    model_path = "microsoft/lora-base"
    user_input = input("Enter your prompt: ")
    num_runs = int(input("Enter the number of inference runs (default is 1): ") or 1)
    optimize_and_infer(model_path, user_input, num_runs)
