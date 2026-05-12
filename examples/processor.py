import chromadb, cv2, torch, clip
from PIL import Image

torch.set_num_threads(2)
torch.set_num_interop_threads(2)

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/16", device=device)
video = cv2.VideoCapture("/app/videos/output_fixed.mkv")
client = chromadb.HttpClient(host="chromadb", port=8000)
collection = client.get_or_create_collection(name="videos")

count = 0

while video.isOpened():
    ret, frame = video.read()

    if not ret:
        break

    count += 1

    if count % 30 != 0:
        continue

    timestamp = count / 30

    print(f"Processing img: {timestamp}")

    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    with torch.no_grad():
        embeddings = model.encode_image(preprocess(img).unsqueeze(0).to(device))

    embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
    embeddings = embeddings.cpu().numpy().tolist()

    collection.add(
        ids=[str(int(timestamp))],
        embeddings=embeddings,
        metadatas=[{"timestamp": timestamp}]
    )
