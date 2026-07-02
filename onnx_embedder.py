import os
import numpy as np
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import AutoTokenizer
import onnxruntime as ort


class ONNXEmbedder:

    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        self.session = ort.InferenceSession(
            f"{model_path}/model.onnx",
            providers=["CPUExecutionProvider"]
        )

    def encode(self, texts, batch_size=32):

        if isinstance(texts, str):
            texts = [texts]

        all_embeddings = []

        expected_inputs = {
            inp.name for inp in self.session.get_inputs()
        }

        for i in range(0, len(texts), batch_size):

            batch = texts[i:i + batch_size]

            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="np"
            )

            # ✅ FIX 1: dtype fix
            inputs = {k: v.astype(np.int64) for k, v in inputs.items()}

            model_inputs = {}

            if "input_ids" in expected_inputs:
                model_inputs["input_ids"] = inputs["input_ids"]

            if "attention_mask" in expected_inputs:
                model_inputs["attention_mask"] = inputs["attention_mask"]

            if (
                "token_type_ids" in expected_inputs
                and "token_type_ids" in inputs
            ):
                model_inputs["token_type_ids"] = inputs["token_type_ids"]

            outputs = self.session.run(None, model_inputs)

            token_embeddings = outputs[0]  # (B, T, H)

            attention_mask = inputs["attention_mask"]  # (B, T)

            # ✅ FIX 2: correct mask shape
            mask = attention_mask[:, :, None].astype(np.float32)

            # pooling
            embeddings = (token_embeddings * mask).sum(axis=1) / (mask.sum(axis=1) + 1e-9)

            # normalization
            embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9

            all_embeddings.append(embeddings)

        return np.vstack(all_embeddings)