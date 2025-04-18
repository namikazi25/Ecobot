import open_clip
import torch
from PIL import Image
import io

# Load BioTrove-CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:BGLab/BioTrove-CLIP')
tokenizer = open_clip.get_tokenizer('hf-hub:BGLab/BioTrove-CLIP')
model.to(device).eval()

def classify_with_biotrove(image_bytes: bytes):
    """Run BioTrove-CLIP for zero-shot classification without predefined labels."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = preprocess_val(image).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        predicted_species = model(image_features)  # Let the model infer species directly

    return predicted_species
