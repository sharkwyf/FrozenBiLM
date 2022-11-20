import os
import torch
import torch.nn
import torch.optim
import numpy as np
import cv2
import argparse
from pathlib import Path

from tqdm import tqdm
from model import build_model, get_tokenizer
from args import get_args_parser
from model.mineclip import MineCLIP, utils as U
from util.misc import get_mask
from util.verb_noun import ALL_WORDS, MINECRAFT_VERBS, ALL_NOUNS, ALL_VERBS



"""
Benchmark Nouns and Verbs Recoginition
"""
@torch.no_grad()
def benchmark_evaluate(model, tokenizer, data, answer_bias_dict, args, device="cpu"):
# questions
    questions = {
        "nouns": [
            ("i see the [MASK]", ""),
            ("i find the [MASK]", ""),
            ("i'm watching the [MASK]", ""),
            ("i'm looking at the [MASK]", ""),
            ("the [MASK] is before me", ""),
            ("the [MASK] is in front of me", ""),
        ],
        "verbs": [
            ("i am [MASK]", "present"),
            ("i am just [MASK]", "present"),
            ("i was [MASK]", "present"),
            ("i was just [MASK]", "present"),
            ("what i'm doing is i'm just [MASK]", "present"),
            ("what i was doing is i was just [MASK]", "present"),
        ],
    }

    # eval
    result = {type: {} for type in list(data.keys())}
    for type, items in data.items():
        print("evaluating", type)

        answer_bias = answer_bias_dict[type]
        texts = [tokenizer(
            question,
            add_special_tokens=True,
            max_length=args.max_tokens,
            truncation=True,
            return_tensors="pt",
        ) for question, phase in questions[type]]

        for key, features in tqdm(items.items()):
            # print(key)
            result[type][key] = []
            for video in features:
                video = video.to(device)
                video_len = torch.tensor(video.size(1), device=device)
                video_mask = get_mask(video_len, video.size(1)).to(device)

                for num, encoded in enumerate(texts):
                    input_ids = encoded["input_ids"].to(device)
                    indices = ((input_ids[0] == tokenizer.mask_token_id) * torch.arange(input_ids.shape[1], device=device)).nonzero()
                    if len(indices) == 0:
                        break
                    min_idx = indices[0]

                    phase = questions[type][num][1]
                    answer = key
                    if phase == "past":
                        answer = MINECRAFT_VERBS[answer][0]
                    elif phase == "present":
                        answer = MINECRAFT_VERBS[answer][1]
                    
                    key_encoded = tokenizer(answer)["input_ids"]
                    if len(key_encoded) > 3:
                        print(answer, key_encoded)
                    key_id = key_encoded[1]

                    # forward
                    output = model(
                        video=video,
                        video_mask=video_mask,
                        input_ids=input_ids,
                        attention_mask=encoded["attention_mask"].to(device),
                    )
                    logits = output.logits[:,video_len:,:len(answer_bias)] + answer_bias
                    encoded_output = logits.argmax(dim=2)

                    # generate one word at a time
                    prediction = tokenizer.decode(encoded_output[0, min_idx])
                    prob = logits.softmax(dim=2)[0, min_idx, key_id]
                    result[type][key].append((answer, prediction, prediction == answer, prob))
                pass
    
    info = {}
    ret = {}
    for type, items in result.items():
        info[type] = {}
        for key, res in items.items():
            total_cnt = sum([1 for x in res])
            total_correct = sum([x[2] for x in res])

            info[type][key] = {
                "acc": total_correct / total_cnt,
                "conf": (sum([x[3] for x in res]) / total_cnt).item()
            }
        
        avg_acc = sum([res["acc"] for res in info[type].values()]) / len(info[type])
        avg_conf = sum([res["conf"] for res in info[type].values()]) / len(info[type])
        print(info[type])
        print("{}: prediction accuracy: {:.3f}%, average confidence: {:.8f}%".format(type, avg_acc * 100, avg_conf * 100))
        ret[type] = {
            "prediction accuracy": avg_acc,
            "average confidence": avg_conf,
        }
    
    print("Benchmark evalution done")
    return ret

@torch.no_grad()
def main(args):
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    # load frozenbilm model
    model = build_model(args)
    model.to(device)
    model.eval()
    tokenizer = get_tokenizer(args)

    # encoded available words
    answer_bias_dict = {}
    for type, words in [("nouns", ALL_NOUNS), ("verbs", ALL_VERBS)]:
        answer_id = tokenizer.encode(["▁" + w for w in list(words)])[1:-1]
        answer_bias = torch.zeros(tokenizer.vocab_size, dtype=torch.float32, device=device)
        answer_bias += args.answer_bias_weight
        for id in answer_id:
            answer_bias[id] = 0
        answer_bias_dict[type] = answer_bias

    # Load pretrained checkpoint
    if args.load:
        print("loading from", args.load)
        checkpoint = torch.load(args.load, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=False)

    # Load benchmark features
    data = np.load(args.feature_path, allow_pickle=True).item()
    
    benchmark_evaluate(model, tokenizer, data, answer_bias_dict, args, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    parser.add_argument("--feature_path", default="./workspace/FrozenBiLM/data/Minedojo/features.npy", type=str)
    parser.add_argument("--answer_bias_weight", default=-100, type=float)
    args = parser.parse_args()
    args.model_name = os.path.join(os.environ["TRANSFORMERS_CACHE"], args.model_name)
    main(args)
