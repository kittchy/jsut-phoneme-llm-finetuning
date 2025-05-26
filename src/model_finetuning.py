from unsloth import FastLanguageModel, FastModel
import torch
from datasets import load_dataset

from trl import SFTTrainer
from transformers.training_args import TrainingArguments
from unsloth import is_bfloat16_supported
from torchmetrics.text import CharErrorRate
from tqdm import tqdm
import yaml

MAX_LENGTH = 2048


def build_model(load_in_4bit: bool = True):
    load_in_8bit = load_in_4bit is False
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/llama-3-8b-bnb-4bit",  # "unsloth/Qwen3-30B-A3B",
        max_seq_length=MAX_LENGTH,  # Choose any for long context!
        load_in_4bit=load_in_4bit,  # 4 bit quantization to reduce memory
        load_in_8bit=load_in_8bit,  # [NEW!] A bit more accurate, uses 2x memory
        full_finetuning=False,  # [NEW!] We have full finetuning now!
        # token = "hf_...", # use one if using gated models
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
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    return model, tokenizer


def build_trainer(model, tokenizer, dataset, max_seq_length):
    return SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=2,  # 2
            gradient_accumulation_steps=4,  # 4
            warmup_steps=5,
            max_steps=100000,
            learning_rate=2e-4,  # 2e-4
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs",
            num_train_epochs=3,
        ),
    )


alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

instructions = """
あなたは、日本語音声合成の専門家です。以下の入力は、JSUTコーパスに基づく音素列であり、韻律記号（^、$、_、#、[、]、?）が含まれています。この記号は文の構造やイントネーション、アクセントを表しています。
あなたのタスクは、この音素列と韻律情報から、元の自然な日本語の文章を正確に復元することです。

■ 入力の説明:

- 音素（例：k a N j i）: 日本語の発音を表します
- 記号:
    - ^: 文の開始
    - $: 文の終わり
    - _: ポーズ（句読点や息継ぎ）
    - #: アクセント句の区切り
    - [: アクセントが低音から高音へ上昇する位置
    - ]: アクセントの核（高音から低音へ下がる位置）
    - ?: 疑問文の語尾（上昇調）

それでは、次の音素列を日本語テキストに変換してください：
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


def evaluate_model(model, tokenizer, dataset, filename: str, max_length=MAX_LENGTH):
    FastLanguageModel.for_inference(model)
    targets = []
    preds = []
    result = {}
    device = next(model.parameters()).device  # モデルのデバイスを取得
    for data in tqdm(dataset):
        targets.append(data["text"])
        inputs = tokenizer(
            [
                alpaca_prompt.format(
                    instructions,
                    data["phoneme"],
                    "",
                )
            ],
            return_tensors="pt",
        ).to("cuda")
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
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # "### Response:" 以降だけ抽出
        response = decoded.split("### Response:")[-1].strip()
        preds.append(response)
        result[data["key"]] = {
            "phoneme": data["phoneme"],
            "text": data["text"],
            "text_pred": response,
        }
    cer = CharErrorRate()
    cer_value = float(cer(preds, targets))
    print("CharErrorRate:", cer_value)

    with open(f"{filename}_cer__{cer_value}.yaml", "w") as f:
        yaml.dump(result, f, allow_unicode=True, default_flow_style=False)


def main():
    ### prepare ###
    train_json = "./train.json"
    test_json = "./test.json"
    model, tokenizer = build_model()
    datasets = build_dataset(tokenizer, train_json)
    test_datasets = build_dataset(tokenizer, test_json)

    evaluate_model(model, tokenizer, test_datasets, "baseline", max_length=MAX_LENGTH)

    print("================ training time ==============")
    #### build trainer ###
    trainer = build_trainer(model, tokenizer, datasets, max_seq_length=MAX_LENGTH)
    trainer.train()
    model.save_pretrained("phoneme-llama-3-8b")

    print("================ evaluation time ==============")

    evaluate_model(model, tokenizer, test_datasets, "finetuning", max_length=MAX_LENGTH)


if __name__ == "__main__":
    main()
