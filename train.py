# Callable file to train the neural model stored in model.py.
# Will create log files at the end of the training and a weight files, result of the training in order to be used later on.
# Training results will be logged using classical prints.
# Created by Gaëtan MARGUERITTE, model inspired by the PhD thesis of Mr. Luyen Le Ngoc.

import pickle
import atexit
import argparse
import sys
import os
import time
import random
import numpy as np
import torch
import pltpublish as pub
from torch.autograd import Variable
import tqdm
import matplotlib.pyplot as plt
from typing import List, Dict
from model import Tagger
from utils import (
    load_dataset,
    load_words_embedding_file,
    build_vocab,
    enhance_vocabulary,
    pad_sequence,
    shuffle_and_split,
    get_data_from_dataset,
    PAD_TAG,
)


def evaluate(
    model: Tagger,
    data: List[Dict[str, List[str]]],
    best_acc: float,
    log_result: bool = False,
):
    def get_batch(batch_index: int, dataset: List):
        dataset_index = batch_index * batch_size
        batch = dataset[dataset_index : dataset_index + batch_size]
        (
            words_batch_text,
            words_batch,
            lemmas_batch,
            postags_batch,
            deprels_batch,
            ccgs_batch,
        ) = ([], [], [], [], [], [])
        masks_batch = []
        for sentence in batch:
            masks_batch.append(
                pad_sequence(
                    [1 for _ in range(len(sentence["sentence_text"]))],
                    max_sequence_length,
                    0,
                )
            )
            words_batch_text.append(sentence["sentence_text"])
            words_batch.append(
                pad_sequence(
                    sentence["words_id"], max_sequence_length, words2id[PAD_TAG]
                )
            )
            lemmas_batch.append(
                pad_sequence(
                    sentence["lemmas_id"], max_sequence_length, lemmas2id[PAD_TAG]
                )
            )
            postags_batch.append(
                pad_sequence(
                    sentence["postags_id"], max_sequence_length, postags2id[PAD_TAG]
                )
            )
            deprels_batch.append(
                pad_sequence(
                    sentence["deprels_id"], max_sequence_length, deprels2id[PAD_TAG]
                )
            )
            ccgs_batch.append(
                pad_sequence(sentence["ccgs_id"], max_sequence_length, ccgs2id[PAD_TAG])
            )
        masks_batch = torch.tensor(masks_batch, device=device, dtype=torch.long)
        words_batch = torch.tensor(words_batch, device=device, dtype=torch.long)
        lemmas_batch = torch.tensor(lemmas_batch, device=device, dtype=torch.long)
        postags_batch = torch.tensor(postags_batch, device=device, dtype=torch.long)
        deprels_batch = torch.tensor(deprels_batch, device=device, dtype=torch.long)
        return (
            words_batch_text,
            words_batch,
            lemmas_batch,
            postags_batch,
            deprels_batch,
            ccgs_batch,
            masks_batch,
        )

    prediction = []
    correct = 0
    total = 0
    batch_index = 0
    number_batch_per_epoch = int(np.ceil(len(data) / batch_size))
    errors: Dict[int, int] = {}
    with torch.no_grad():
        for _ in range(number_batch_per_epoch):
            words_text, words, lemmas, postags, deprels, ccgs, masks = get_batch(
                batch_index, data
            )
            batch_index += 1
            predictions = model(words, lemmas, postags, deprels, masks)
            for i in range(len(predictions)):
                for (word, predicted_tag, true_tag) in zip(
                    words_text[i], predictions[i], ccgs[i]
                ):
                    # print(f"pred vs truth for [{id2words[word]}] = {predicted_tag}, {true_tag} \t ({val[i]})")
                    line = " ".join([word, id2ccgs[predicted_tag], id2ccgs[true_tag]])
                    prediction.append(line)
                    if predicted_tag == true_tag:
                        correct += 1
                    elif log_result:
                        errors[true_tag] = (
                            errors[true_tag] + 1 if true_tag in errors else 1
                        )
                    total += 1
                prediction.append("")
        errorsf = eval_temp + "/unrecognized_tags.txt"
        predf = eval_temp + "/pred.model"
        with open(predf, "w") as f:
            f.write("\n".join(prediction))
        if log_result:
            with open(errorsf, "w") as f:
                f.write("\n".join([f"{id2ccgs[x]} \t\t {errors[x]}" for x in errors]))
            print(f"Errors logged in {errorsf}")
    new_acc = correct / total
    print(f"Correct / wrong guesses: {correct} / {total-correct} ")
    if new_acc > best_acc:
        best_acc = new_acc

    return new_acc, best_acc


def train_model():
    def save_model() -> None:
        torch.save(model.state_dict(), model_file)
        print(f"Saved model dictionary state file in {model_file}")

    accuracy_train, best_accuracy_train = evaluate(model, data_train, 0)
    accuracy_valid, best_accuracy_valid = evaluate(model, data_valid, 0)
    print("Accuracy:")
    print(f"\t- train: {accuracy_train}")
    print(f"\t- valid: {accuracy_valid}")

    def do_batch(batch_index: int, dataset: List) -> int:
        dataset_index = batch_index * batch_size
        batch = dataset[dataset_index : dataset_index + batch_size]
        words_batch, lemmas_batch, postags_batch, deprels_batch, ccgs_batch = (
            [],
            [],
            [],
            [],
            [],
        )
        masks_batch = []
        for sentence in batch:
            masks_batch.append(
                pad_sequence(
                    [1 for _ in range(len(sentence["sentence_text"]))],
                    max_sequence_length,
                    0,
                )
            )
            words_batch.append(
                pad_sequence(
                    sentence["words_id"], max_sequence_length, words2id[PAD_TAG]
                )
            )
            lemmas_batch.append(
                pad_sequence(
                    sentence["lemmas_id"], max_sequence_length, lemmas2id[PAD_TAG]
                )
            )
            postags_batch.append(
                pad_sequence(
                    sentence["postags_id"], max_sequence_length, postags2id[PAD_TAG]
                )
            )
            deprels_batch.append(
                pad_sequence(
                    sentence["deprels_id"], max_sequence_length, deprels2id[PAD_TAG]
                )
            )
            ccgs_batch.append(
                pad_sequence(sentence["ccgs_id"], max_sequence_length, ccgs2id[PAD_TAG])
            )
        masks_batch = torch.tensor(masks_batch, device=device, dtype=torch.long)
        words_batch = torch.tensor(words_batch, device=device, dtype=torch.long)
        lemmas_batch = torch.tensor(lemmas_batch, device=device, dtype=torch.long)
        postags_batch = torch.tensor(postags_batch, device=device, dtype=torch.long)
        deprels_batch = torch.tensor(deprels_batch, device=device, dtype=torch.long)
        ccgs_batch = torch.tensor(ccgs_batch, device=device, dtype=torch.long)
        # gradient descent
        model.zero_grad()
        nll = model.nll(
            words_batch,
            lemmas_batch,
            postags_batch,
            deprels_batch,
            ccgs_batch,
            masks_batch,
        )
        nll.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        return nll.item()

    lr = 0.015
    optimizer = torch.optim.RMSprop(
        model.parameters(), lr=lr, weight_decay=1e-4, momentum=0.9
    )
    lr_reduce = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        patience=1,
        factor=0.2,
        verbose=True,
        threshold_mode="abs",
        threshold=1e-3,
    )

    loss = 0.0
    history = []
    best_accuracy_train = -1.0
    best_accuracy_valid = -1.0
    data_train_len = len(data_train)
    print("Starting training...")
    model.train(True)
    number_batch_per_epoch = int(np.ceil(len(data_train) / batch_size))
    total_counter = 0
    atexit.register(save_model)
    for i in range(epochs):
        print(f"Epoch {i+1}/{epochs}")
        random.shuffle(data_train)
        batch_index = 0
        for _ in tqdm.trange(number_batch_per_epoch, desc="batchs"):
            loss += do_batch(batch_index, data_train)
            total_counter += batch_size
            batch_index += 1

        model.train(False)
        accuracy_train, best_accuracy_train = evaluate(
            model, data_train, best_accuracy_train
        )
        previous_best = best_accuracy_valid
        accuracy_valid, best_accuracy_valid = evaluate(
            model, data_valid, best_accuracy_valid
        )
        if previous_best != best_accuracy_valid:
            save_model()
        print("Accuracy:")
        print(f"\t- train: {accuracy_train}")
        print(f"\t- valid: {accuracy_valid}")
        model.train(True)
        avg_loss = loss / number_batch_per_epoch
        history.append(avg_loss)
        loss = 0.0
        print(f"Average batch loss during this epoch: \t{avg_loss}")
        lr_reduce.step(round(best_accuracy_valid, 4))
    atexit.unregister(save_model)
    pub.setup()
    plt.figure()
    plt.plot(history)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    pub.save_fig("loss.png")
    plt.show()
    plt.close()
    accuracy_test, best_accuracy_test = evaluate(model, data_test, -1.0, True)
    print(f"Final test accuracy: {accuracy_test}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="French CCG supertagger [train].")
    parser.add_argument(
        "-m", "--model", default="model.pt", type=str, help="model file"
    )
    parser.add_argument(
        "-ep",
        "--epochs",
        default=60,
        type=int,
        help="Number of training epochs (default: 60)",
    )
    parser.add_argument(
        "-em",
        "--embed",
        default="data/frWac_non_lem_no_postag_no_phrase_200_skip_cut100.bin",
        type=str,
        help="binary word embedding file from word2vec (default: data/frWac_non_lem_no_postag_no_phrase_200_skip_cut100.bin)",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        default="data/ccgresult28.txt",
        type=str,
        help="dataset file of FrenchCCGBank in txt format (default: data/ccgresult28.txt)",
    )
    parser.add_argument(
        "-vr",
        "--validation-ratio",
        default=0.2,
        type=float,
        help="validation/training ratio for training (default: 0.2)",
    )
    parser.add_argument(
        "-tr",
        "--test-ratio",
        default=0.2,
        type=float,
        help="test/dataset ratio for training (default: 0.2)",
    )
    parser.add_argument(
        "-s", "--seed", default=0, type=int, help="random seed (default: 0)"
    )
    parser.add_argument(
        "-we",
        "--word-embedding",
        default=200,
        type=int,
        help="word embedding size (default: 200)",
    )
    parser.add_argument(
        "-b", "--batch-size", default=16, type=int, help="batch size (default: 16)"
    )
    parser.add_argument(
        "--max-sequence-length",
        default=120,
        type=int,
        help="Max sequence length (default: 120)",
    )
    parser.add_argument(
        "--hidden-size", default=128, type=int, help="Hidden size (default: 128)"
    )

    parameters = parser.parse_args()
    model_file: str = parameters.model
    # output_file: str = parameters.output
    epochs: int = parameters.epochs
    dataset_file: str = parameters.dataset
    embedding_file: str = parameters.embed
    vratio: float = parameters.validation_ratio
    tratio: float = parameters.test_ratio
    seed: int = parameters.seed
    word_embedding_dim: int = parameters.word_embedding
    batch_size: int = parameters.batch_size
    max_sequence_length: int = parameters.max_sequence_length
    hidden_size: int = parameters.hidden_size

    # Get device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    eval_path = "./evaluation"
    eval_temp = os.path.join(eval_path, "temp")
    start_time = time.time()
    if not os.path.exists(dataset_file) or not os.path.isfile(dataset_file):
        print("Dataset must be a valid dataset file!", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(embedding_file) or not os.path.isfile(embedding_file):
        print("Embed must be a valid pre words embedding file!", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(eval_temp):
        os.makedirs(eval_temp)
    # vec2vec embedding fetch
    pre_words_embedding = load_words_embedding_file(embedding_file)
    # dataset loading
    dataset = load_dataset(dataset_file)

    # vocabs is a list of pairs of dictionary. First element is item2id, second elemnt is id2item.
    # item2id, id2item = vocabs[i]
    # where i = 0 for words, 1 for lemmas, 2 for postags, 3 for deprel and 4 for ccgs
    vocabs = build_vocab(dataset)
    # some words might be in pretrained words embedding but not in vocab.
    words2id, id2words = vocabs[0]
    words2id, id2words = enhance_vocabulary(words2id, id2words, pre_words_embedding)
    vocabs[0] = (words2id, id2words)
    lemmas2id, id2lemmas = vocabs[1]
    postags2id, id2postags = vocabs[2]
    deprels2id, id2deprels = vocabs[3]
    ccgs2id, id2ccgs = vocabs[4]

    # prepare word embedding with vocabulary ids
    word_embeds = np.random.uniform(
        -np.sqrt(0.06), np.sqrt(0.06), (len(words2id), word_embedding_dim)
    )
    for word in words2id:
        if word in pre_words_embedding:
            word_embeds[words2id[word]] = pre_words_embedding[word]

    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # shuffling and splitting dataset
    dataset_train, dataset_valid, dataset_test = shuffle_and_split(
        dataset, vratio, tratio, seed
    )
    print(
        f"Lengths of training / validation / test datasets in number of sentences: {len(dataset_train)} / {len(dataset_valid)} / {len(dataset_test)}"
    )

    data_train = get_data_from_dataset(
        dataset_train, words2id, lemmas2id, postags2id, deprels2id, ccgs2id
    )
    data_valid = get_data_from_dataset(
        dataset_valid, words2id, lemmas2id, postags2id, deprels2id, ccgs2id
    )
    data_test = get_data_from_dataset(
        dataset_test, words2id, lemmas2id, postags2id, deprels2id, ccgs2id
    )

    # model initialization
    model = Tagger(
        word_embeds,
        len(lemmas2id),
        len(postags2id),
        len(deprels2id),
        len(ccgs2id),
        ccgs2id,
        len(words2id),
        max_sequence_length=max_sequence_length,
        batch_size=batch_size,
        hidden_size=hidden_size,
    )
    model.to(device)

    # training

    end_time = time.time()
    print(f"Training prep done in {end_time - start_time}s")

    start_time = time.time()
    train_model()
    end_time = time.time()
    print(f"Training done in {end_time - start_time}s")

    pickle_file = "model_data.pickle"
    print(f"Saving model data in {pickle_file}...")

    model_data = {
        "words2id": words2id,
        "id2words": id2words,
        "lemmas2id": lemmas2id,
        "id2lemmas": id2lemmas,
        "postags2id": postags2id,
        "id2postags": id2postags,
        "deprels2id": deprels2id,
        "id2deprels": id2deprels,
        "ccgs2id": ccgs2id,
        "id2ccgs": id2ccgs,
        "lemma_embed_size": len(lemmas2id),
        "postag_embed_size": len(postags2id),
        "deprel_embed_size": len(deprels2id),
        "nb_output_class": len(ccgs2id),
        "max_sequence_length": max_sequence_length,
        "batch_size": batch_size,
        "hidden_size": hidden_size,
        "vocab_size": len(words2id),
    }

    with open(pickle_file, "wb") as f:
        pickle.dump(model_data, f, protocol=pickle.HIGHEST_PROTOCOL)
