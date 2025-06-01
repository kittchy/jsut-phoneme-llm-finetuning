from unsloth import FastLanguageModel
import torch
from datasets import load_dataset

from trl import SFTTrainer
from transformers.training_args import TrainingArguments
from transformers import EarlyStoppingCallback
from torchmetrics.text import CharErrorRate
from tqdm import tqdm
import yaml

MAX_LENGTH = 512


def build_model(load_in_4bit: bool = True, load_in_8bit: bool = False):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen3-4B-unsloth-bnb-4bit",  # "unsloth/Qwen3-30B-A3B",
        max_seq_length=MAX_LENGTH,  # Choose any for long context!
        load_in_4bit=load_in_4bit,  # 4 bit quantization to reduce memory
        load_in_8bit=load_in_8bit,  # [NEW!] A bit more accurate, uses 2x memory
        full_finetuning=False,  # [NEW!] We have full finetuning now!
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # 8, 16, 32, 64, 128
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    return model, tokenizer


def build_trainer(model, tokenizer, train_dataset, eval_dataset, max_seq_length):
    return SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=16,
        packing=False,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        args=TrainingArguments(
            fp16_full_eval=True,
            per_device_eval_batch_size=2,
            eval_accumulation_steps=4,
            per_device_train_batch_size=4,  # 2
            gradient_accumulation_steps=8,  # 4
            report_to="wandb",
            run_name="unsloth-llm-finetuning-with-qwen3",
            warmup_steps=5,
            num_train_epochs=3,  # 3
            eval_steps=10,
            eval_on_start=True,
            eval_strategy="steps",
            do_eval=True,
            learning_rate=2e-4,  # 2e-4
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
        ),
    )


alpaca_prompt = """{}

Input: {}
Output: {}
"""
instructions = """
あなたは、日本語音声合成の専門家です。以下の入力は、JSUTコーパスに基づく音素列であり、韻律記号（^、$、_、#、[、]、?）が含まれています。この記号は文の構造やイントネーション、アクセントを表しています。
あなたのタスクは、この音素列と韻律情報から、元の自然な日本語の文章を正確に復元することです。

Inputの説明:

- 音素（例：k a N j i）: 日本語の発音を表します
- 記号:
    - ^: 文の開始
    - $: 文の終わり
    - _: ポーズ（句読点や息継ぎ）
    - #: アクセント句の区切り
    - [: アクセントが低音から高音へ上昇する位置
    - ]: アクセントの核（高音から低音へ下がる位置）
    - ?: 疑問文の語尾（上昇調）

Outputの説明：
- 音素列から適切か書き起こした結果を出力してください

"""


def build_dataset(tokenizer, dataset_json):
    EOS_TOKEN = tokenizer.eos_token

    def formatting_prompts_func(examples):
        inputs = examples["phoneme"]
        outputs = examples["text"]
        # key = examples["key"]
        instructions_list = [instructions] * len(inputs)
        texts = []
        for instruction, input, output in zip(instructions_list, inputs, outputs):
            text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
            texts.append(text)
        return {"prompt": texts}

    dataset = load_dataset("json", data_files=dataset_json, split="train")
    dataset = dataset.map(formatting_prompts_func, batched=True)
    return dataset


def evaluate_model(
    model, tokenizer, dataset, filename: str, max_length=MAX_LENGTH, batch_size=16
):
    FastLanguageModel.for_inference(model)
    targets = []
    preds = []
    result = {}
    device = next(model.parameters()).device  # モデルのデバイスを取得

    # datasetをリスト化
    data_list = list(dataset)
    total = len(data_list)
    for i in tqdm(range(0, total, batch_size)):
        batch = data_list[i : i + batch_size]
        batch_targets = [data["text"] for data in batch]
        batch_keys = [data["key"] for data in batch]
        batch_phonemes = [data["phoneme"] for data in batch]
        batch_prompts = [
            alpaca_prompt.format(
                instructions,
                phoneme,
                "",
            )
            for phoneme in batch_phonemes
        ]
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # 推論
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_length,
                do_sample=False,
            )
        # デコード
        decoded_list = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        # "### Response:" 以降だけ抽出
        responses = [decoded.split("Output: ")[-1].strip() for decoded in decoded_list]
        preds.extend(responses)
        for key, phoneme, text, response in zip(
            batch_keys, batch_phonemes, batch_targets, responses
        ):
            result[key] = {
                "phoneme": phoneme,
                "text": text,
                "text_pred": response,
            }
            print(f"phoneme: {phoneme}")
            print(f"text: {text}")
            print(f"text_pred: {response}")
            print("===")
        targets.extend(batch_targets)

    cer = CharErrorRate()
    cer_value = float(cer(preds, targets))
    print("CharErrorRate:", cer_value)

    with open(f"{filename}_cer__{cer_value}.yaml", "w") as f:
        yaml.dump(result, f, allow_unicode=True, default_flow_style=False)


def main():
    ### prepare ###
    train_json = "./train.json"
    eval_json = "./eval.json"
    test_json = "./test.json"
    model, tokenizer = build_model()
    train_datasets = build_dataset(tokenizer, train_json)
    eval_datasets = build_dataset(tokenizer, eval_json)
    test_datasets = build_dataset(tokenizer, test_json)

    # evaluate_model(model, tokenizer, test_datasets, "baseline", max_length=MAX_LENGTH)

    print("================ training time ==============")
    #### build trainer ###
    trainer = build_trainer(
        model, tokenizer, train_datasets, eval_datasets, max_seq_length=MAX_LENGTH
    )

    trainer.train()
    model.save_pretrained("phoneme-llama-3-8b")

    print("================ evaluation time ==============")

    evaluate_model(model, tokenizer, test_datasets, "finetuning", max_length=MAX_LENGTH)


if __name__ == "__main__":
    main()
