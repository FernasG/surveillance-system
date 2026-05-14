import chromadb

client = chromadb.HttpClient(host="localhost", port=8000)

collections = client.list_collections()

for c in collections:
    print(c.name)

    # collection = client.get_collection(c.name)

    # client.delete_collection(c.name)
    # data = collection.get()

    # print(data)