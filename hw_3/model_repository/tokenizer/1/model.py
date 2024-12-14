from transformers import AutoTokenizer
import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        """
        Initialize the model by loading the tokenizer.
        """
        model_path = "/assets/tokenizer"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.max_length = 128  # Maximum sequence length

    def execute(self, requests):
        """
        Process input requests and return tokenized outputs.
        """
        responses = []
        for request in requests:
            # Extract input text
            input_texts = pb_utils.get_input_tensor_by_name(request, "TEXT").as_numpy()
            
            # Input is already a numpy array of byte strings
            decoded_texts = [text.decode("utf-8") for text in input_texts.flatten()]

            # Tokenize the text
            tokenized = self.tokenizer(
                decoded_texts,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="np"
            )

            # Prepare outputs
            input_ids = tokenized["input_ids"].astype(np.int64)
            attention_mask = tokenized["attention_mask"].astype(np.int64)

            # Create Triton output tensors
            input_ids_tensor = pb_utils.Tensor("INPUT_IDS", input_ids)
            attention_mask_tensor = pb_utils.Tensor("ATTENTION_MASK", attention_mask)

            # Add to responses
            responses.append(pb_utils.InferenceResponse(output_tensors=[input_ids_tensor, attention_mask_tensor]))

        return responses

    def finalize(self):
        """
        Finalize and clean up resources if necessary.
        """
        pass

