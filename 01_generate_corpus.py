# 01_generate_corpus.py
import json, random

TOPICS = ["transformer architecture", "graph neural networks",
          "reinforcement learning", "computer vision", "NLP"]
METHODS = ["we propose", "this paper introduces", "we present",
           "our approach leverages", "we demonstrate"]
RESULTS = ["outperforms baselines by {:.1f}%".format(random.uniform(1,15)),
           "achieves state-of-the-art results", "reduces error by {:.1f}%".format(random.uniform(5,30))]

def generate_abstract(i):
    topic = random.choice(TOPICS)
    method = random.choice(METHODS)
    result = random.choice(RESULTS)
    return {
        "id": i,
        "title": f"Paper {i}: {topic.title()} Study",
        "abstract": f"{method.capitalize()} a novel approach to {topic}. "
                    f"Using attention mechanisms and {random.choice(['contrastive learning','data augmentation','pruning'])}, "
                    f"our model {result} on standard benchmarks."
    }

corpus = [generate_abstract(i) for i in range(2000)]
with open("corpus/abstracts.json", "w") as f:
    json.dump(corpus, f, indent=2)