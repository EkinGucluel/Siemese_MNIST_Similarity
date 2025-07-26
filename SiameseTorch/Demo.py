import gradio as gr # type: ignore
import torch
from torchvision import datasets, transforms
from PIL import Image
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from PIL import ImageOps

# ========== Your Siamese Model Definition ==========
class SiameseNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(512)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.embedding = nn.Linear(512, 64)

        self.fc1 = nn.Linear(128, 64)
        self.out = nn.Linear(64, 1)

    def cnn(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.embedding(x)
        return x

    def forward(self, imgA, imgB):
        featA = self.cnn(imgA)
        featB = self.cnn(imgB)
        x = torch.cat([featA, featB], dim=1)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.out(x))
        return x

# ========== Setup ==========
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = SiameseNetwork().to(device)
model.load_state_dict(torch.load("siamese_model_epoch20_bs8_train200_testval50_20250723.pth", map_location=device))
model.eval()

# Load MNIST test dataset
mnist_test = datasets.MNIST(root="./data", train=False, download=True)

# Build index lists for each digit label
indices_by_label = {i: [] for i in range(10)}
for idx, (_, lbl) in enumerate(mnist_test):
    indices_by_label[lbl].append(idx)

to_tensor = transforms.Compose([
    transforms.Grayscale(),  # convert canvas to 1 channel
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])



def preprocess_user(pil_image):
    # 1) ensure grayscale
    image = pil_image.convert("L")
    # 2) invert to match MNIST format
    image = ImageOps.invert(image)
    # 3) tensor+normalize
    image = to_tensor(image).unsqueeze(0)
    return image.to(device)

def preprocess_mnist(pil_image):
    # MNIST is already white-on-black, so no invert
    image = pil_image.convert("L")
    image = to_tensor(image).unsqueeze(0)
    return image.to(device)

# ========== New Predict Function ==========
# ========== Predict & Sample Function ==========
def predict_all_digits(user_drawing):
    # extract raw array
    if isinstance(user_drawing, dict):
        user_drawing = user_drawing["composite"]

    user_img    = Image.fromarray(user_drawing)
    user_tensor = preprocess_user(user_img)

    images = []
    scores = {}
    for digit in range(10):
        # pick a random index from that digit’s pool
        idx = random.choice(indices_by_label[digit])
        mn_img, _ = mnist_test[idx]

        mn_tensor = preprocess_mnist(mn_img)
        with torch.no_grad():
            score = model(user_tensor, mn_tensor).item()

        images.append(np.array(mn_img))
        scores[str(digit)] = round(score, 4)

    return images, scores

# ========== Gradio UI ==========
with gr.Blocks() as demo:
    gr.Markdown("## ✍️ Draw a digit\nSee similarity to all MNIST digits 0–9")

    with gr.Row():
        with gr.Column(scale=1):
            canvas = gr.Sketchpad(
                label="Draw Here",
                image_mode="L",
                width=280, height=280
            )
            btn_compare = gr.Button("Compare to MNIST 0–9")
            btn_refresh = gr.Button("Refresh Samples (0–9)")
        with gr.Column(scale=1):
            gallery     = gr.Gallery(label="MNIST Digits", columns=1)
            json_scores = gr.JSON(label="Similarity Scores")

    # both buttons trigger the same sampling+scoring
    for btn in (btn_compare, btn_refresh):
        btn.click(
            fn=predict_all_digits,
            inputs=[canvas],
            outputs=[gallery, json_scores]
        )

demo.launch()