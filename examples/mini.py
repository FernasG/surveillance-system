import cv2, torch, mobileclip
from PIL import Image

model, _, preprocess = mobileclip.create_model_and_transforms('mobileclip_s0', pretrained='/root/.cache/huggingface/hub/models--apple--MobileCLIP-S0/snapshots/71aa3e13dda93115871afbd017336535ba29886c/mobileclip_s0.pt')
tokenizer = mobileclip.get_tokenizer('mobileclip_s0')
torch.set_num_threads(2)
cv2.setNumThreads(0)

# image = preprocess(Image.open("/app/app/CLIP.png").convert('RGB')).unsqueeze(0)
# text = tokenizer(["a diagram", "a dog", "a cat"])

# with torch.no_grad(), torch.cuda.amp.autocast():
#     image_features = model.encode_image(image)
#     text_features = model.encode_text(text)
#     image_features /= image_features.norm(dim=-1, keepdim=True)
#     text_features /= text_features.norm(dim=-1, keepdim=True)

#     text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

# print("Label probs:", text_probs)

video = cv2.VideoCapture("/app/videos/output_fixed.mkv")

video_features = []
timestamps = []

count = 0

while video.isOpened():
    ret, frame = video.read()

    if not ret:
        break

    if count % 30 == 0:
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        with torch.no_grad():
            feature = model.encode_image(preprocess(img).unsqueeze(0))
            video_features.append(feature)
            timestamps.append(count / 30)
    
        break

print(len(video_features))
print(len(timestamps))
print(count)

query_text = "a blue car"

# with torch.no_grad():
#     text_feature = model.encode_text(tokenizer([query_text]))

# similarities = [torch.cosine_similarity(text_feature, f) for f in video_features]
# melhor_momento = timestamps[similarities.index(max(similarities))]

# print(f"O trecho que você busca provavelmente está no segundo: {melhor_momento}")