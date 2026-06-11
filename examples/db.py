import chromadb
import json

client = chromadb.HttpClient(host="localhost", port=8000)

collections = client.list_collections()

for c in collections:
    print(c.name)

    collection = client.get_collection(c.name)

    print(f"Number of items: {collection.count()}")

    data = collection.get(include=["embeddings"])

    # json_data = json.dumps(data, indent=2)

    print(data)
    # print(json_data)