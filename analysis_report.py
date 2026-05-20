import numpy as np
from embedder import text_to_embedding
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
import pickle
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib.patches import FancyBboxPatch
import networkx as nx

with open('vectors_2d.pkl', 'rb') as f:
    words, vectors_2d = pickle.load(f)
    

df_words = pd.DataFrame({
    '№': range(1, len(words)+1),
    'Слово': words,
    'Длина': [len(w) for w in words]
})
print(df_words.to_string(index=False))

distances = pdist(vectors_2d, metric='cosine')
distance_matrix = squareform(distances)

dist_df = pd.DataFrame(distance_matrix, index=words, columns=words)
print("\nМатрица косинусных расстояний (0=идентичны, 1=противоположны):")
print(dist_df.round(4))

pairs = []
n = len(words)
for i in range(n):
    for j in range(i+1, n):
        pairs.append({
            'Слово 1': words[i],
            'Слово 2': words[j],
            'Расстояние': distance_matrix[i, j],
            'Сходство': 1 - distance_matrix[i, j]
        })

df_pairs = pd.DataFrame(pairs)
df_pairs_sorted = df_pairs.sort_values('Сходство', ascending=False)


def semantic_search(query, vectors, words, top_k=5):
    query_vec = text_to_embedding(query)
    similarities = [1 - np.linalg.norm(query_vec - vec) / 2 for vec in vectors]
    top_indices = np.argsort(similarities)[::-1][:top_k]
    return [(words[i], similarities[i]) for i in top_indices]

test_queries = words
full_vectors = np.array([text_to_embedding(w) for w in words])

for query in test_queries:
    results = semantic_search(query, full_vectors, words, top_k=4)
    print(f"\n Запрос: '{query}'")
    df_results = pd.DataFrame(results, columns=['Слово', 'Сходство'])
    print(df_results.to_string(index=False))

plt.figure(figsize=(12, 8))
plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], s=100, alpha=0.7)
for i, word in enumerate(words):
    plt.annotate(word, (vectors_2d[i, 0], vectors_2d[i, 1]), fontsize=10)
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("Визуализация эмбеддингов слов")
plt.grid(True, alpha=0.3)
plt.show()


fig = plt.figure(figsize=(18, 12))

ax1 = fig.add_subplot(2, 3, 1)

G = nx.Graph()
for i, word in enumerate(words):
    G.add_node(word, pos=vectors_2d[i])

distances = pdist(full_vectors, metric='cosine')
n_words = len(words)
dist_list = []
for i in range(n_words):
    for j in range(i+1, n_words):
        dist_list.append((i, j, distances[len(dist_list)]))

dist_list.sort(key=lambda x: x[2])
for i, j, dist in dist_list[:15]:  
    G.add_edge(words[i], words[j], weight=1-dist)

pos = {word: vectors_2d[i] for i, word in enumerate(words)}
nx.draw(G, pos, ax=ax1, node_color='lightblue', node_size=800, 
        font_size=9, font_weight='bold', edge_color='gray', 
        alpha=0.7, with_labels=True)
ax1.set_title("Граф семантических связей\n(топ-15 самых похожих пар)", fontsize=12)


filename2 = "visualizations/graph_semantic_links.png"
plt.savefig(filename2, dpi=300, bbox_inches='tight')
plt.close()
