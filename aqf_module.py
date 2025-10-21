import torch
import open_clip
from PIL import Image

# 1️⃣ CLIP 모델 불러오기
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
tokenizer = open_clip.get_tokenizer('ViT-B-32')

# 2️⃣ 이미지 불러오기
image = preprocess(Image.open("dog.jpg")).unsqueeze(0)

# 3️⃣ 쿼리 문장 여러개 생성 (query variants)
queries = [
    "a dog running on the grass",
    "a brown dog playing outdoors",
    "a dog sprinting in the field",
    "a running puppy outside",
    "brown dog running fast"
]

# 4️⃣ 각 문장을 Text Encoder 통과시켜 임베딩
text_embeds = []
with torch.no_grad():
    for q in queries:
        tok = tokenizer([q])
        e_i = model.encode_text(tok)
        e_i = e_i / e_i.norm(dim=-1, keepdim=True)
        text_embeds.append(e_i)

E = torch.cat(text_embeds, dim=0)  # (5, 512)

# 5️⃣ 이미지 임베딩
with torch.no_grad():
    image_emb = model.encode_image(image)
    image_emb = image_emb / image_emb.norm(dim=-1, keepdim=True)

# 6️⃣ Baseline: 단순 평균
fusion_mean = E.mean(dim=0)
fusion_mean = fusion_mean / fusion_mean.norm()

# 7️⃣ AQF: Adaptive Query Fusion (유사도 기반 softmax 가중)
sims = (E @ image_emb.T).squeeze(1)              # 각 query별 유사도
weights = torch.softmax(sims / 0.05, dim=0)      # 온도 0.05
fusion_aqf = (weights.unsqueeze(1) * E).sum(dim=0)
fusion_aqf = fusion_aqf / fusion_aqf.norm()

# 8️⃣ 결과 비교
sim_mean = (fusion_mean @ image_emb.T).item()
sim_aqf = (fusion_aqf @ image_emb.T).item()

print("🧩 Query별 유사도:", [round(s.item(), 3) for s in sims])
print("⚙️  Softmax 가중치:", [round(w.item(), 3) for w in weights])
print(f"\nBaseline (mean fusion): {sim_mean:.3f}")
print(f"AQF (adaptive fusion):  {sim_aqf:.3f}")
