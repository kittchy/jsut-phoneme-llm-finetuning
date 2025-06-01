import yaml
import json


def load_yaml(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(data, path):
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, sort_keys=False)


def process_phoneme(text):
    text_split = text.split("-")
    return " ".join(text_split[1:-1])


def main():
    phoneme_path = "jsut-label/e2e_symbol/phoneme.yaml"
    text_path = "jsut-label/text_kana/basic5000.yaml"
    train_path = "train.json"
    eval_path = "eval.json"
    test_path = "test.json"

    phoneme_dict = load_yaml(phoneme_path)
    text_dict = load_yaml(text_path)

    result = []
    for key in phoneme_dict:
        if key in text_dict:
            text = text_dict[key].get("text_level0", "")
            # phoneme = phoneme_dict[key]
            phoneme = process_phoneme(phoneme_dict[key])

            result.append({"text": text, "phoneme": phoneme, "key": key})

    all_length = 5000
    train = result[: int(all_length * 0.8)]
    eval = result[int(all_length * 0.8) : int(all_length * 0.9)]
    test = result[int(all_length * 0.9) :]

    with open(train_path, "w", encoding="utf-8") as f:
        for r in train:
            json_str = json.dumps(r, ensure_ascii=False) + "\n"
            f.write(json_str)
    with open(eval_path, "w", encoding="utf-8") as f:
        for r in eval:
            json_str = json.dumps(r, ensure_ascii=False) + "\n"
            f.write(json_str)
    with open(test_path, "w", encoding="utf-8") as f:
        for r in test:
            json_str = json.dumps(r, ensure_ascii=False) + "\n"
            f.write(json_str)


if __name__ == "__main__":
    main()
