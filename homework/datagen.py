import json
import math

from .cot import CoTModel
from .data import Dataset


def generate_dataset(output_json: str, oversample: int = 10, temperature: float = 0.6):
    trainset = Dataset("train")
    model = CoTModel("HuggingFaceTB/SmolLM2-1.7B-Instruct")

    questions = [q for q, _ in trainset]
    answers = [float(a) for _, a in trainset]

    all_generations = model.batched_generate(
        [model.format_prompt(q) for q in questions],
        num_return_sequences=oversample,
        temperature=temperature,
    )

    rft_data = []

    for question, true_answer, generations in zip(questions, answers, all_generations):
        chosen_reasoning = None

        for gen in generations:
            pred = model.parse_answer(gen)

            if math.isnan(pred):
                continue

            if abs(pred - true_answer) <= max(1e-4, 0.05 * abs(true_answer)):
                chosen_reasoning = gen
                break

        if chosen_reasoning is not None:
            rft_data.append([question, true_answer, chosen_reasoning])

    with open(output_json, "w") as f:
        json.dump(rft_data, f, indent=2)

    print(f"Saved {len(rft_data)} examples to {output_json}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"generate": generate_dataset})
