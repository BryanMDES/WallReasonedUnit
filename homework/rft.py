from .base_llm import BaseLLM
from .sft import test_model


def load() -> BaseLLM:
    from pathlib import Path
    from peft import PeftModel

    model_name = "rft_model"
    model_path = Path(__file__).parent / model_name

    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()

    return llm


def train_model(
    output_dir: str,
    **kwargs,
):
    import json
    from pathlib import Path

    from peft import LoraConfig, get_peft_model
    from transformers import Trainer, TrainingArguments

    from .sft import tokenize

    class RFTDataset:
        def __init__(self, tokenizer, data):
            self.tokenizer = tokenizer
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            question, answer, reasoning = self.data[idx]
            return tokenize(
                self.tokenizer,
                question=question,
                answer=reasoning,
            )

    data_path = Path(__file__).parent.parent / "data" / "rft.json"
    with open(data_path, "r") as f:
        rft_data = json.load(f)

    llm = BaseLLM()

    lora_config = LoraConfig(
        r=16,
        lora_alpha=64,
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
    )

    llm.model = get_peft_model(llm.model, lora_config)

    if llm.device == "cuda":
        llm.model.enable_input_require_grads()

    llm.model.config.use_cache = False

    tokenized_trainset = RFTDataset(llm.tokenizer, rft_data)

    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=output_dir,
        report_to="tensorboard",
        learning_rate=1e-4,
        per_device_train_batch_size=32,
        num_train_epochs=7,
        gradient_checkpointing=True,
        save_strategy="epoch",
        logging_steps=10,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=llm.model,
        args=training_args,
        train_dataset=tokenized_trainset,
    )

    trainer.train()
    trainer.save_model(output_dir)

    test_model(output_dir)


if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})