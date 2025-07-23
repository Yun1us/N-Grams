🧠 NLP Homework 03 – N-Gram Language Modeling

This project builds Unigram and Bigram language models using the Penn Treebank dataset.

Included Features:
	•	Tokenization with <STOP> marker
	•	Unigram and Bigram counting
	•	Log-probability and Perplexity computation
	•	Replacing rare words with <unk>
	•	Add-One (Laplace) Smoothing
	•	Model comparison on test set

⸻

🔧 Quick Start

# 1. (Optional but recommended) Create virtual environment
python -m venv .venv
source .venv/bin/activate      # on macOS/Linux
.venv\Scripts\activate         # on Windows

# 2. Install dependencies
pip install datasets numpy tqdm

# 3. Run the script
python NLP_03.py
