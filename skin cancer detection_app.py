import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
class BinaryMobileNet(nn.Module):
    def __init__(self):
        super(BinaryMobileNet, self).__init__()
        self.model = models.mobilenet_v2(pretrained=False)
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Linear(in_features, 1)
        )
    def forward(self, x):
        return self.model(x)
@st.cache_resource
def load_model():
    model = BinaryMobileNet()
    model.load_state_dict(torch.load("mobilenet_skin_binary.pth", map_location=torch.device('cpu')))
    model.eval()
    return model
model = load_model()
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)])
st.title("Skin Cancer Detection")
st.write("Upload a skin image to check for cancerous signs.")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_column_width=True)
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(img_tensor)
        prob = torch.sigmoid(output).item()
        prediction = "Cancer Detected" if prob >= 0.5 else "No Cancer Detected"
        st.write(f"### Prediction: {prediction}")
        st.write(f"Confidence: {prob:.4f}")
