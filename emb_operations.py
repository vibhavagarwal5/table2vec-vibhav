import argparse
import os
import numpy as np
from MulticoreTSNE import MulticoreTSNE as TSNE
import umap
import matplotlib.pyplot as plt
import torch

from utils import loadpkl


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path",
                        help="path for the scores")
    parser.add_argument("--model",
                        help="model file")
    parser.add_argument("--vocab_path",
                        help="vocab file")
    return parser.parse_args()


def dim_red_plot(plt_type, emb, vocab, output_dir, n_components=2, random_state=42):
    print(f"-- Start {plt_type} --")
    if plt_type == 'tsne':
        new_values = TSNE(
            n_components=n_components,
            random_state=random_state,
            n_jobs=10,
            verbose=2
        ).fit_transform(emb)
        x = []
        y = []
        for value in new_values:
            x.append(value[0])
            y.append(value[1])

    elif plt_type == 'umap':
        new_values = umap.UMAP(
            n_components=n_components,
            random_state=random_state
        ).fit_transform(emb)
        x, y = new_values[:, 0], new_values[:, 1]

    print("-- Start ploting --")
    plt.figure(figsize=(16, 16))
    plt.scatter(x, y)
    # for i in range(len(x)):
    #     plt.annotate(vocab[i], xy=(x[i], y[i]), xytext=(
    #         5, 2), textcoords="offset points", ha="right", va="bottom")
    plt.savefig(os.path.join(output_dir, f'viz/emb_{plt_type}.png'))


if __name__ == "__main__":
    args = get_args()
    model = torch.load(os.path.join(args.path, args.model))
    vocab = loadpkl(args.vocab_path)
    emb = model['embeddings.weight'].cpu().data.numpy()
    dim_red_plot('umap', emb, vocab, args.path, 3)

    # e1 = torch.load(
    #     './output/5_20_22_0_39/model_1.pt')['embeddings.weight'].cpu().data
    # e2 = torch.load(
    #     './output/5_20_22_0_39/model_0.pt')['embeddings.weight'].cpu().data
    # print(e1.shape, e2.shape)
    # print(torch.equal(e1,e2))
    # print(torch.sum(torch.sub(e1, e2)))
