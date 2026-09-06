import chromadb
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="company_handbook")

data = collection.get(include=["embeddings", "documents"])
vectors = data["embeddings"]
docs = data["documents"]

pca = PCA(n_components=2)
points_2d = pca.fit_transform(vectors)

plt.figure(figsize=(10, 8))
plt.scatter(points_2d[:, 0], points_2d[:, 1])

for i, doc in enumerate(docs):
    label = doc[:30].replace("\n", " ") + "..."
    plt.annotate(label, (points_2d[i, 0], points_2d[i, 1]), fontsize=8)

plt.title("Vector không gian ngữ nghĩa (PCA 2D)")
plt.savefig("vector_plot.png")
plt.show()