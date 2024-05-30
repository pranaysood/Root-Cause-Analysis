import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import pinecone
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')

file_path = '/content/combined_data (1).csv'
data = pd.read_csv(file_path)

data['textual_representation'] = data['textual_representation'].fillna('').astype(str)

api_key = '13b1255d-5993-4098-ae44-2b1e0cd62e22'
pc = Pinecone(api_key=api_key)
index_name = "synthetic-data-meta"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=model.get_sentence_embedding_dimension(),
        metric="cosine",
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )

# Directly use the Index class to interact with the index
index = pc.Index(name=index_name)

# Batch processing setup
batch_size = 100
batches = (len(data) + batch_size - 1) // batch_size

# Store embeddings for visualization
all_embeddings = []

print("Starting batch processing and upsert...")

for batch_num in tqdm(range(batches)):
    start_index = batch_num * batch_size
    end_index = min((batch_num + 1) * batch_size, len(data))
    batch_data = data.iloc[start_index:end_index]

    # Encode batch data
    embeddings = model.encode(batch_data['textual_representation'].tolist())
    all_embeddings.extend(embeddings)

    # Collect metadata and prepare vectors for upsert
    vectors = []
    for i, row in batch_data.iterrows():
        metadata = {col: row[col] for col in data.columns if col != 'textual_representation'}
        vectors.append((str(i), embeddings[i - start_index], metadata))
    
    # Upsert batch data into Pinecone
    index.upsert(vectors=vectors)

    print(f"Batch {batch_num + 1}/{batches} upserted.")

print("All data processed and upserted successfully.")

# Perform PCA for visualization of embeddings
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(np.array(all_embeddings))

# Plotting the reduced embeddings
plt.figure(figsize=(10, 10))
plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], alpha=0.5)
plt.title('PCA of Embeddings')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()
