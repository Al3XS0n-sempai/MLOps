import torch
from transformers import AutoModel, AutoTokenizer
from fvcore.nn import FlopCountAnalysis, parameter_count_table


MODEL_NAME = "cointegrated/rubert-tiny-toxicity"

def classify_layers(flops_analysis, memory_bandwidth, batch_size_threshold):
    compute_bound = []
    memory_bound = []

    for layer, stats in flops_analysis.items():
        memory_usage_bytes = stats / memory_bandwidth

        flops_per_byte = stats / memory_usage_bytes if memory_usage_bytes > 0 else 0

        if flops_per_byte > batch_size_threshold:
            compute_bound.append(layer)
        else:
            memory_bound.append(layer)

    return compute_bound, memory_bound

def compute_flops(model_name, input_text, device="cuda", memory_bandwidth=760e9, batch_size_threshold=38):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    model_inputs = (input_ids, attention_mask)

    def print_in_box(text: str):
        print(f"|{text:^98}|")

    with torch.no_grad():
        flops = FlopCountAnalysis(model, model_inputs)

        # FLOPS per layer
        flops_by_layer = flops.by_module()
        print(f"{'':-^100}")
        print(f"FLOPs per layer:")
        print(f"{'':-^100}")
        for layer, stats in flops_by_layer.items():
            print_in_box(f"Layer {layer}: {stats}FLOPs")
        print(f"{'':-^100}")
        print("\n\n")


        #TOTALL FLOPSS
        print(f"{'':-^100}")
        print(f"Total FLOPs: {flops.total()}")
        print(f"{'':-^100}")
        print("\n\n")

        # PARAMETERS COUNT
        print(f"{'':-^100}")
        print("Parameter count:\n", parameter_count_table(model))
        print(f"{'':-^100}")
        print("\n\n")


        compute_bound, memory_bound = classify_layers(
            flops_by_layer, memory_bandwidth, batch_size_threshold
        )

        # Compute-bound layers
        print(f"{'':-^100}")
        print(f"Compute-bound layers:\n{compute_bound}")
        print(f"{'':-^100}")
        print("\n\n")

        # MEMORY bound layser
        print(f"{'':-^100}")
        print(f"Memory-bound layers:\n{memory_bound}")
        print(f"{'':-^100}")


if __name__ == "__main__":
    model_name = MODEL_NAME
    input_text = "Говно текст вопрос говна"
    compute_flops(model_name, input_text)
