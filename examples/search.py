import chromadb, torch, clip

device = "cuda" if torch.cuda.is_available() else "cpu"
model, _ = clip.load("ViT-B/16", device=device)

client = chromadb.HttpClient(host="chromadb", port=8000)
collection = client.get_collection("videos")

query_text = "a red car"

with torch.no_grad():
    query_embeddings = model.encode_text(clip.tokenize([query_text]).to(device))

query_embeddings = query_embeddings / query_embeddings.norm(dim=-1, keepdim=True)
query_embeddings = query_embeddings.cpu().numpy().tolist()

# results = collection.get()

results = collection.query(
    query_embeddings=query_embeddings,
    n_results=3
)

print(results)
