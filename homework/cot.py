from .base_llm import BaseLLM


class CoTModel(BaseLLM):
    def format_prompt(self, question: str) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful unit conversion assistant. "
                    "Solve the problem step by step very briefly. "
                    "Always end with the final result inside <answer></answer> tags."
                ),
            },
            {
                "role": "user",
                "content": "How many grams are there in 2 kg?",
            },
            {
                "role": "assistant",
                "content": (
                    "1 kg = 1000 grams. "
                    "So 2 kg = 2 * 1000 = 2000 grams. "
                    "<answer>2000</answer>"
                ),
            },
            {
                "role": "user",
                "content": question,
            },
        ]

        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

def load() -> CoTModel:
    return CoTModel()


def test_model():
    from .data import Dataset, benchmark

    testset = Dataset("valid")
    model = CoTModel()
    benchmark_result = benchmark(model, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"test": test_model, "load": load})
