import torch
import cv2
import numpy as np
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity

# Initialize ResNet18 model for Re-ID
def get_feature_extractor():
    from torchvision import models
    resnet = models.resnet18(pretrained=True)
    resnet = torch.nn.Sequential(*list(resnet.children())[:-1])  # Remove last FC layer
    resnet.eval()
    return resnet

# Transform image crop for feature extraction
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Extract feature from cropped image
def extract_feature(crop, model):
    if crop.size == 0:
        return None
    inp = transform(crop).unsqueeze(0)
    with torch.no_grad():
        feature = model(inp).squeeze().numpy()
    return feature

# Match feature to existing players using cosine similarity
def assign_id(feature, players, used_ids, frame_no, sim_threshold=0.8):
    best_match = None
    best_score = sim_threshold

    for pid, pdata in players.items():
        if pid in used_ids:
            continue
        sim = cosine_similarity([feature], [pdata['feature']])[0][0]
        if sim > best_score:
            best_match = pid
            best_score = sim

    if best_match is None:
        return None
    else:
        players[best_match]['feature'] = feature
        players[best_match]['last_seen'] = frame_no
        return best_match
