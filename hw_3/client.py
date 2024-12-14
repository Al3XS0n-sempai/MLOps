import numpy as np
import tritonclient.grpc as grpcclient
import sys


def call_triton(input_text):
    """
        Triton сервер и возвращает 5 эмбеддингов: 4 от TensorRT и 1 от ONNX.
    """
    try:
        triton_client = grpcclient.InferenceServerClient(url="localhost:8101", verbose=False)

        input_texts = np.array([[input_text.encode("utf-8")]])
        inputs = [grpcclient.InferInput("TEXT", input_texts.shape, "BYTES")]
        inputs[0].set_data_from_numpy(input_texts)

        outputs = [
            grpcclient.InferRequestedOutput("LOGITS_FP16"),
            grpcclient.InferRequestedOutput("LOGITS_FP32"),
            grpcclient.InferRequestedOutput("LOGITS_INT8"),
            grpcclient.InferRequestedOutput("LOGITS_BEST"),
            grpcclient.InferRequestedOutput("LOGITS_ONNX"),
        ]

        response = triton_client.infer(
            model_name="ensemble",
            inputs=inputs,
            outputs=outputs,
        )

        embeddings = {
            "tensorrt": [
                response.as_numpy("LOGITS_FP16"),
                response.as_numpy("LOGITS_FP32"),
                response.as_numpy("LOGITS_INT8"),
                response.as_numpy("LOGITS_BEST"),
            ],
            "onnx":  response.as_numpy("LOGITS_ONNX"),
        }

        return embeddings
    except Exception as e:
        print(f"Ошибка при вызове Triton: {e}")
        sys.exit(1)


def check_quality(input_text):
    """
        Сравнивает отклонения эмбеддингов TensorRT с эталонным ONNX эмбеддингом.
        Возвращает средние отклонения.
    """
    embeddings = call_triton(input_text)

    tensorrt_embeddings = embeddings["tensorrt"]
    onnx_embedding = embeddings["onnx"]

    deviations = []

    for tensorrt_embedding in tensorrt_embeddings:
        deviation = np.linalg.norm(tensorrt_embedding - onnx_embedding)
        deviations.append(deviation)

    mean_deviation = np.mean(deviations)

    return mean_deviation


def main():
    texts = [
        "это простой текст",
        "негр тупой ты",
        "вонучая какашка",
        "ты мразь",
        "просто добрый текст",
    ]

    total_deviation = 0
    for text in texts:
        deviation = check_quality(text)
        total_deviation += deviation
        print(f"Текст: {text}, Отклонение: {deviation:.6f}")

    average_deviation = total_deviation / len(texts)
    print(f"Среднее отклонение для всех текстов: {average_deviation:.6f}")


if __name__ == "__main__":
    main()

