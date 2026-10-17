"""
Hosts a local language model:
    lm = LanguageModel("qwen2.5-7b")
    lm.forward("Caption this meme template")

    # with a system prompt
    lm.forward("Caption this meme template", system_prompt="You write short, punchy captions.")

    # multi-turn: user_prompts[i] is answered by assistant_answers[i];
    # the last user prompt has no answer yet (that's what the model replies to)
    lm.forward(
        user_prompts=["Hi", "Give me a meme idea", "Make it about Mondays"],
        assistant_answers=["Hello!", "A cat that refuses to wake up."],
        system_prompt="You are a meme expert.",
    )

Requirements: pip install torch transformers accelerate
"""
import torch
from transformers import (AutoModelForCausalLM, AutoProcessor, AutoTokenizer, 
                        Gemma3ForConditionalGeneration, LlavaNextForConditionalGeneration, LlavaNextProcessor,)

SUPPORTED_MODELS = {
    "qwen2.5-7b": ("Qwen/Qwen2.5-7B-Instruct", "causal"),
    "llama-3.1-8b": ("meta-llama/Llama-3.1-8B-Instruct", "causal"),
    "gemma-3-12b": ("google/gemma-3-12b-it", "gemma3"),
    "llava-1.6-7b": ("llava-hf/llava-v1.6-mistral-7b-hf", "llava"),
}


class LanguageModel:
    def __init__(self, model_name: str, max_new_tokens: int = 512, temperature: float = 0.0):
        key = model_name.lower()
        if key not in SUPPORTED_MODELS:
            raise ValueError(
                f"Unknown model '{model_name}'. Choose from: {list(SUPPORTED_MODELS)}"
            )

        self.model_name = key
        self.repo_id, self.kind = SUPPORTED_MODELS[key]
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._load()

    def _load(self):
        def load(model_cls):
            try:
                return model_cls.from_pretrained(
                    self.repo_id, dtype=torch.bfloat16, device_map="auto"
                )
            except TypeError:
                return model_cls.from_pretrained(
                    self.repo_id, torch_dtype=torch.bfloat16, device_map="auto"
                )

        if self.kind == "causal":  # Qwen2.5, Llama 3.1
            self.tokenizer = AutoTokenizer.from_pretrained(self.repo_id)
            self.processor = None
            self.model = load(AutoModelForCausalLM)
        elif self.kind == "gemma3":
            self.processor = AutoProcessor.from_pretrained(self.repo_id)
            self.tokenizer = self.processor.tokenizer
            self.model = load(Gemma3ForConditionalGeneration)
        elif self.kind == "llava":
            self.processor = LlavaNextProcessor.from_pretrained(self.repo_id)
            self.tokenizer = self.processor.tokenizer
            self.model = load(LlavaNextForConditionalGeneration)
        self.model.eval()

    def format_messages(self, user_prompts, assistant_answers=None, system_prompt=None) -> list:
        """Interleave into [system, user, assistant, user, assistant, ..., user].
        len(user_prompts) must equal len(assistant_answers) + 1, because the
        final user turn is the one the model is about to answer.
        """
        if isinstance(user_prompts, str):
            user_prompts = [user_prompts]
        assistant_answers = list(assistant_answers or [])

        if len(user_prompts) != len(assistant_answers) + 1:
            raise ValueError(
                f"Need exactly one more user prompt than assistant answers, got "
                f"{len(user_prompts)} user and {len(assistant_answers)} assistant."
            )

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        for user, assistant in zip(user_prompts, assistant_answers):
            messages.append({"role": "user", "content": user})
            messages.append({"role": "assistant", "content": assistant})
        messages.append({"role": "user", "content": user_prompts[-1]})
        return messages

    def _to_structured(self, messages: list) -> list:
        """Gemma 3 / LLaVA templates expect content as [{'type': 'text', 'text': ...}]."""
        return [
            {"role": m["role"], "content": [{"type": "text", "text": m["content"]}]}
            for m in messages
        ]


    def _fold_system_into_user(self, messages: list) -> list:
        """LLaVA's Mistral template has no system role: prepend it to the first user turn."""
        if messages and messages[0]["role"] == "system":
            system, rest = messages[0], [dict(m) for m in messages[1:]]
            rest[0]["content"] = f"{system['content']}\n\n{rest[0]['content']}"
            return rest
        return messages

    def forward(self, user_prompts, assistant_answers=None, system_prompt=None) -> str:
        """Return the model's reply to the last user prompt."""
        import torch

        messages = self.format_messages(user_prompts, assistant_answers, system_prompt)

        # Render the chat template to a string, then tokenize with the plain
        # tokenizer. This works the same for all four models, text-only,
        # without needing an image for the LLaVA / Gemma processors.
        if self.kind == "causal":
            template_owner, template_messages = self.tokenizer, messages
        else:
            if self.kind == "llava":
                messages = self._fold_system_into_user(messages)
            template_owner, template_messages = self.processor, self._to_structured(messages)

        text = template_owner.apply_chat_template(
            template_messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        inputs = inputs.to(self.model.device)

        gen_kwargs = dict(
            max_new_tokens=self.max_new_tokens,
            do_sample=self.temperature > 0,
            pad_token_id=self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.tokenizer.eos_token_id,
        )
        if self.temperature > 0:
            gen_kwargs["temperature"] = self.temperature

        with torch.inference_mode():
            output = self.model.generate(**inputs, **gen_kwargs)

        new_tokens = output[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    __call__ = forward