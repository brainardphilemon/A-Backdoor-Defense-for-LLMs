import torch
import argparse
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer
from PackDataset import packDataset_util_bert


def read_data(file_path):
    import pandas as pd
    data = pd.read_csv(file_path, sep='\t').values.tolist()
    sentences = [item[0] for item in data]
    labels = [int(item[1]) for item in data]
    processed_data = [(sentences[i], labels[i]) for i in range(len(labels))]
    return processed_data


def precalculate_loo_scores(model, tokenizer, data, layer_idx=6, desc=""):
   
    model.eval()
    results = []

    backbone = model.module.bert if hasattr(model, "module") else model.bert

    with torch.no_grad():
        for sent, label in tqdm(data, desc=desc):
            split_sent = sent.split()
            if len(split_sent) > 0 and split_sent[-1] == "":
                split_sent = split_sent[:-1]

            num_words = len(split_sent)
            base_sent = " ".join(split_sent)

           
            sentences = [base_sent]
            for i in range(num_words):
                new_sent = " ".join(split_sent[:i] + split_sent[i+1:])
                sentences.append(new_sent)

            inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

           
            outputs = backbone(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[layer_idx]
            clss = hidden_states[:, 0, :]  

          
            logits = model(**inputs)[0]
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            base_cls = clss[0]
            new_clss = clss[1:]

            base_pred = preds[0]
            new_preds = preds[1:]

            if new_clss.shape[0] > 0:
                shifts = torch.norm(
                    base_cls.expand_as(new_clss) - new_clss,
                    p=2,
                    dim=1
                ).cpu().numpy()

                m = np.median(shifts)
                mad = np.median(np.abs(shifts - m))
                epsilon = 1e-6
                z_scores = (shifts - m) / (1.4826 * mad + epsilon)
            else:
                z_scores = np.array([])

            results.append({
                "label": label,
                "base_sent": base_sent,
                "base_pred": base_pred,
                "z_scores": z_scores,
                "new_preds": new_preds,
                "words": split_sent
            })

    return results


def remove_all_outliers(sentence_words, z_scores, tau):

    if len(sentence_words) == 0:
        return ""

    keep_mask = np.abs(z_scores) <= tau
    kept_words = [w for w, keep in zip(sentence_words, keep_mask) if keep]

    
    return " ".join(kept_words)


def build_filtered_dataset(cached_data, tau, is_poison=False, target_label=1):

    processed_data = []

    for item in cached_data:
        words = item["words"]
        z_scores = item["z_scores"]

        filtered_sent = remove_all_outliers(words, z_scores, tau)

        if is_poison:
            processed_data.append((filtered_sent, target_label))
        else:
            processed_data.append((filtered_sent, item["label"]))

    return processed_data


def evaluate_loader(model, loader):
    model.eval()
    total_correct = 0
    total_number = 0

    with torch.no_grad():
        for padded_text, attention_masks, labels in loader:
            if torch.cuda.is_available():
                padded_text = padded_text.cuda()
                attention_masks = attention_masks.cuda()
                labels = labels.cuda()

            output = model(padded_text, attention_masks)[0]
            _, idx = torch.max(output, dim=1)

            total_correct += (idx == labels).sum().item()
            total_number += labels.size(0)

    return total_correct / total_number if total_number > 0 else 0.0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='sst-2')
    parser.add_argument('--model_path', default='')
    parser.add_argument('--clean_data_path', default='')
    parser.add_argument('--poison_data_path', default='')
    parser.add_argument('--target_label', default=1, type=int)
    parser.add_argument('--layer_idx', default=6, type=int)
    parser.add_argument('--record_file', default='filtered_outliers.log')
    parser.add_argument('--batch_size', default=32, type=int)
    args = parser.parse_args()

    model = torch.load(args.model_path, weights_only=False)
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    packDataset_util = packDataset_util_bert()

    f = open(args.record_file, 'w')

    
    poison_data = read_data(args.poison_data_path)
    poison_data = [(sent, args.target_label) for sent, label in poison_data]

    clean_data = read_data(args.clean_data_path)

    
    print("Precalculating scores for poisoned dataset...")
    poison_cached = precalculate_loo_scores(
        model, tokenizer, poison_data, layer_idx=args.layer_idx, desc="Poison Data"
    )

    print("Precalculating scores for clean dataset...")
    clean_cached = precalculate_loo_scores(
        model, tokenizer, clean_data, layer_idx=args.layer_idx, desc="Clean Data"
    )

   
    taus = np.arange(0, 15, 0.25)

    for tau in taus:
        print(f"Evaluating tau = {tau}")

        filtered_poison = build_filtered_dataset(
            poison_cached, tau=tau, is_poison=True, target_label=args.target_label
        )
        filtered_clean = build_filtered_dataset(
            clean_cached, tau=tau, is_poison=False
        )

        poison_loader = packDataset_util.get_loader(
            filtered_poison, shuffle=False, batch_size=args.batch_size
        )
        clean_loader = packDataset_util.get_loader(
            filtered_clean, shuffle=False, batch_size=args.batch_size
        )

        attack_success_rate = evaluate_loader(model, poison_loader)
        clean_acc = evaluate_loader(model, clean_loader)

        print('tau: ', tau, file=f)
        print('attack success rate: ', attack_success_rate, file=f)
        print('clean acc: ', clean_acc, file=f)
        print('*' * 89, file=f)

    f.close()
    print("Done!")