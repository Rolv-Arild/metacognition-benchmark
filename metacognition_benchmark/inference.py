"""Local model inference using HuggingFace transformers."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


class LocalModel:
    """Loads a HuggingFace model for local inference with full logprob access."""

    def __init__(self, model_name: str, device: str = "auto", dtype=None, quantization: str = None):
        """
        Args:
            model_name: HuggingFace model ID or local path.
            device: Device map for model placement ("auto", "cpu", "cuda:0", etc.)
            dtype: Torch dtype (default: bfloat16).
            quantization: Quantization mode — None, "4bit", or "8bit".
        """
        if dtype is None:
            dtype = torch.bfloat16
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        load_kwargs = {"device_map": device}

        if quantization == "4bit":
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        elif quantization == "8bit":
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        else:
            load_kwargs["torch_dtype"] = dtype

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        self.model.eval()

    @property
    def device(self):
        return self.model.device

    def get_logprobs_for_completion(self, prompt: str, completion: str) -> list[dict]:
        """Run a single forward pass over prompt+completion, return per-token logprobs for the completion part.

        Returns a list of dicts, one per completion token:
            - token: decoded string of the token
            - token_id: integer token id
            - logprob: log probability of this token given the prefix
            - eos_logprob: log probability of the EOS token at this position
            - top_logprobs: dict mapping decoded token -> logprob for top-10 tokens
        """
        full_text = prompt + completion
        inputs = self.tokenizer(full_text, return_tensors="pt").to(self.device)
        prompt_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"]
        prompt_len = prompt_ids.shape[1]

        with torch.no_grad():
            outputs = self.model(**inputs)

        # logits shape: (1, seq_len, vocab_size)
        logits = outputs.logits[0]  # (seq_len, vocab_size)
        log_probs = torch.log_softmax(logits, dim=-1)

        input_ids = inputs["input_ids"][0]
        eos_id = self.tokenizer.eos_token_id

        results = []
        # Position i predicts token at position i+1
        # We want logprobs for completion tokens, which start at prompt_len
        for i in range(prompt_len - 1, len(input_ids) - 1):
            next_token_id = input_ids[i + 1].item()
            token_str = self.tokenizer.decode([next_token_id])
            lp = log_probs[i, next_token_id].item()
            eos_lp = log_probs[i, eos_id].item() if eos_id is not None else None

            # Top-k for analysis
            top_vals, top_ids = log_probs[i].topk(10)
            top_logprobs = {
                self.tokenizer.decode([tid.item()]): tv.item()
                for tid, tv in zip(top_ids, top_vals)
            }

            results.append({
                "token": token_str,
                "token_id": next_token_id,
                "logprob": lp,
                "eos_logprob": eos_lp,
                "top_logprobs": top_logprobs,
            })

        return results

    def generate(self, messages: list[dict], max_tokens: int = 150, temperature: float = 0.0) -> str:
        """Generate a chat completion locally using the model's chat template."""
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        gen_kwargs = {"max_new_tokens": max_tokens, "do_sample": temperature > 0}
        if temperature > 0:
            gen_kwargs["temperature"] = temperature

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

