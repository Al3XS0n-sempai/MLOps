import torch
import onnx
import onnxruntime
from torch import nn
from transformers import AutoModel, AutoTokenizer
import numpy as np

from model_perf import calc_perf_data


MODEL_NAME: str = "cointegrated/rubert-tiny-toxicity"
SIZE_N: int = 816


class RubertTinyToxicityModel(nn.Module):
    def __init__(self, model_name: str = "", N: int = SIZE_N):
        super().__init__()

        self.transformer = AutoModel.from_pretrained(model_name)

        hidden_dim = self.transformer.config.hidden_size
        self.fc = nn.Linear(hidden_dim, N)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        
        logits = self.fc(last_hidden_state)  # понижаем размерность до N
        return logits


def main():
    model_name = MODEL_NAME
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    onnx_provider = 'CUDAExecutionProvider' if torch.cuda.is_available() else 'CPUExecutionProvider'

    if torch.cuda.is_available():
        print(f"{'ЗАПУСК С ИСПОЛЬЗОВАНИЕММ CUDA НА:':^60}")
        print(f"{torch.cuda.get_device_name(torch.cuda.current_device):^60}")
    else:
        print(f"{'CUDA НЕ ДОСТУПНА, ЗАПУСК НА CPU':^60}")
    
    print("-" * 60)

    n = SIZE_N
    model = RubertTinyToxicityModel(model_name=model_name, N=n)
    model.to(device)
    model.eval() # взяли предобученную модель поэтому ставим в EVAL режтим
 
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    text: str = ""
    with open("input.txt", "r") as f:
        text = f.read()

    print(f"ТЕКСТ ДЛЯ ПРИМЕРА:\n{text}")
    print("-" * 60)
    encoding = tokenizer(text, return_tensors='np')
    input_ids = torch.tensor(encoding["input_ids"], dtype=torch.long).to(device)
    attention_mask = torch.tensor(encoding["attention_mask"], dtype=torch.long).to(device)
    
    onnx_path = "rubert_tiny_toxicity.onnx"
    tokenizer_path = "tokenizer/"
    tokenizer.save_pretrained(tokenizer_path)

    torch.onnx.export(
        model,
        (input_ids, attention_mask),
        onnx_path,
        input_names=["INPUT_IDS", "ATTENTION_MASK"],
        output_names=["LOGITS"],
        dynamic_axes={
            "INPUT_IDS": {0: "BATCH_SIZE", 1: "SEQUENCE_LENGTH"},
            "ATTENTION_MASK": {0: "BATCH_SIZE", 1: "SEQUENCE_LENGTH"},
            "LOGITS": {0: "BATCH_SIZE", 1: "SEQUENCE_LENGTH"}
        },
        opset_version=19,
        do_constant_folding=True
    )

    # Санити-чек через onnxruntime
    ort_session = onnxruntime.InferenceSession(onnx_path, providers=[onnx_provider])
    ort_inputs = {
        "INPUT_IDS": input_ids.detach().cpu().numpy(),
        "ATTENTION_MASK": attention_mask.detach().cpu().numpy()
    }

    # Запускаем инференс на GPU (результат вернется на CPU в виде numpy)
    ort_outs = ort_session.run(None, ort_inputs)
    print(f"{'ONNXRUNTIME OUTPUT':^60}")
    print(ort_outs[0])
    print("-" * 60)

    # Получаем выводы PyTorch-модели (на GPU), затем приводим к CPU и numpy
    torch_out = model(input_ids, attention_mask).detach().cpu().numpy()
    print(f"{'ONNXRUNTIME OUTPUT':^60}")
    print(torch_out)
    print("-" * 60)

    # Проверка близости результата
    print(f"{'СРАВНЕНИЕ OUTPUTS':^60}")
    np.testing.assert_allclose(ort_outs[0], torch_out, rtol=1e-3, atol=1e-3)
    print("ВЫХОДЫ ONNX И PyTorch СОВПАДАЮТ.")
    print("-" * 60)

    print(f"{'ОЦЕНКА ПРОИЗВОДИТЕЛЬНОСТИ':^60}")
    calc_perf_data(MODEL_NAME)
    

if __name__ == "__main__":
    main()
