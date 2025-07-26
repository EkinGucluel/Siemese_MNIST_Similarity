# Siemese_MNIST_Similarity

# ==== Siamese Neural Network for MNIST Digit Similarity ====
  An interactive deep learning project that uses a Siamese Network to measure similarity between a user-drawn digit and MNIST digits (0–9). Trained using PyTorch and deployed with a Gradio-powered web interface.

# ==== Features ====

  - Siamese Network: A neural architecture trained to compare two images and output a similarity score.
  - MNIST Dataset: Standard handwritten digits dataset used for training and evaluation.
  - Gradio Interface: Lets users draw their own digit and compare it to randomly sampled MNIST digits.
  - Pair-based Training: Model learns by comparing digit image pairs, not just classifying digits.
  - Similarity Scores: Returns a probability score (0 to 1) for how similar the drawn digit is to each digit from 0–9.
  - Data Augmentation Ready: Setup supports transformations and on-the-fly pairing for extensibility.

# ==== Modules Used ====
  - torch
  - torchvision
  - numpy
  - matplotlib
  - PIL
  - gradio
  - tqdm

# ==== Siamese Network Architecture ====
  - Shared Convolutional Encoder with 4 convolutional layers
  - Adaptive average pooling for fixed-size embeddings
  - Linear embedding layer (512 → 64)
  - Final similarity score computed via MLP on concatenated embeddings
  - Binary cross-entropy used as the loss function during training

# ==== Structure ====
  - Training Pipeline (train.py or main script)
  - Loads a small, balanced subset of MNIST
  - Forms image pairs (positive/negative) with a custom PairedDataset
  - Trains the Siamese model using Binary Cross Entropy
  - Saves model weights after each epoch
  - Gradio UI App (Demo.py)
  - Allows the user to draw a digit
  - Compares the drawn digit to randomly sampled MNIST digits (0–9) and Displays similarity scores and visual comparisons

  # ==== Similarity Testing Inside The Code ====

  <img width="1004" height="580" alt="Screenshot 2025-07-26 at 5 09 55 PM" src="https://github.com/user-attachments/assets/36db12de-cf60-4e16-a228-ad36c8a6045f" />

  # ==== Example Use Case ====

  <img width="2912" height="1832" alt="Opera Snapshot_2025-07-26_171059_127 0 0 1" src="https://github.com/user-attachments/assets/c0eab658-8559-495b-8bdf-be2da88cb0b8" />  
  <img width="2912" height="1832" alt="Opera Snapshot_2025-07-26_171239_127 0 0 1" src="https://github.com/user-attachments/assets/57bf44ef-51e5-455e-a629-d2353b268134" />

