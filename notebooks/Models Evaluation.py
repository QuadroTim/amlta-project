import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

data = {
    "Model": [
        "jina-embeddings-v2-base-de",
        "jina-embeddings-v2-base-de Fine-tuned",
        "multilingual-e5-large",
        "LaBSE",
        "paraphrase-multilingual-mpnet-base-v2"
    ],
    "Precision@10": [0.0813, 0.0813, 0.0813, 0.0375, 0.0563],
    "Recall@10": [0.8125, 0.8125, 0.8125, 0.3750, 0.5625],
    "F1-Score@10": [0.1477, 0.1477, 0.1477, 0.0682, 0.1023],
    "NDCG@10": [0.8125, 0.8125, 0.8125, 0.3750, 0.5625],
    "Hit Rate@10": [0.8125, 0.8125, 0.8125, 0.3750, 0.5625],
    "MRR@10": [0.5325, 0.5646, 0.4575, 0.2083, 0.2806]
}

df = pd.DataFrame(data)

sns.set(style="whitegrid")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))  # 2 строки, 3 столбца графиков
fig.suptitle('Models effectiveness comparison', fontsize=16)

def plot_bar(ax, x, y, title, color, highlight_color="red"):
    best_index = np.argmax(y)

    colors = [color] * len(x)
    colors[best_index] = highlight_color

    bars = ax.bar(x, y, color=colors)
    ax.set_title(title)
    ax.set_xticks(range(len(x)))
    ax.set_xticklabels(x, rotation=45, ha='right')
    ax.set_ylim(0, 1)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

plot_bar(axes[0, 0], df["Model"], df["Precision@10"], "Precision@10", "skyblue")
plot_bar(axes[0, 1], df["Model"], df["Recall@10"], "Recall@10", "lightgreen")
plot_bar(axes[0, 2], df["Model"], df["F1-Score@10"], "F1-Score@10", "salmon")
plot_bar(axes[1, 0], df["Model"], df["NDCG@10"], "NDCG@10", "gold")
plot_bar(axes[1, 1], df["Model"], df["Hit Rate@10"], "Hit Rate@10", "orchid")
plot_bar(axes[1, 2], df["Model"], df["MRR@10"], "MRR@10", "lightcoral")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()



