# Week 5 Deep Learning Application in Data Science
## Fashion-MNIST Clothing Image Classification with a Regularized CNN

This project implements a reproducible convolutional neural network (CNN) in TensorFlow/Keras for classifying 28×28 grayscale clothing images into ten Fashion-MNIST categories.

### Objective 1
Build, train, and evaluate a reproducible deep-learning image-classification pipeline that can distinguish ten categories of clothing images and explain the model's design choices, results, limitations, and opportunities for improvement.

### Why this dataset?
Fashion-MNIST is public, compact, balanced across ten classes, and challenging enough to demonstrate convolution, regularization, data augmentation, validation strategy, and error analysis without requiring a GPU.

### Project structure
```
fashion_mnist_cnn_project/
├── train.py                    # End-to-end training and evaluation script
├── requirements.txt             # Python dependencies
├── README.md                    # Setup and execution instructions
├── .gitignore
├── docs/
│   └── architecture_diagram.png
└── outputs/                    # Created after training
    ├── best_model.keras
    ├── history.csv
    ├── metrics.json
    ├── classification_report.csv
    ├── confusion_matrix.png
    ├── training_curves.png
    └── sample_predictions.png
```

### Recommended environment
- Python 3.10–3.12
- VS Code with the Python extension
- CPU is sufficient; a GPU shortens training time
- At least 2 GB free disk space for the TensorFlow environment and cached dataset

### Setup in VS Code
```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Run the full experiment
```bash
python train.py --epochs 20 --batch-size 128 --output-dir outputs
```

For a quick smoke test:
```bash
python train.py --epochs 2 --batch-size 128 --output-dir outputs_quick
```

The first run downloads Fashion-MNIST through `tf.keras.datasets.fashion_mnist.load_data()`. The script then creates a stratified training/validation split, trains the CNN, saves the best checkpoint, and writes all evaluation artifacts to the selected output directory.

### Reproducibility
The script fixes Python, NumPy, and TensorFlow random seeds and uses a fixed stratified split. Exact results may still vary slightly across operating systems, TensorFlow versions, and hardware.

### Outputs to inspect
1. `metrics.json` — test accuracy, top-3 accuracy, macro precision, macro recall, and macro F1.
2. `classification_report.csv` — class-level precision, recall, F1, and support.
3. `confusion_matrix.png` — which clothing categories are confused.
4. `training_curves.png` — learning curves for diagnosing underfitting/overfitting.
5. `sample_predictions.png` — qualitative examples with predicted probabilities.

### Dataset labels
0. T-shirt/top · 1. Trouser · 2. Pullover · 3. Dress · 4. Coat · 5. Sandal · 6. Shirt · 7. Sneaker · 8. Bag · 9. Ankle boot

### Academic note
The report includes an expected-results interpretation and the code is the source of truth for measured results. After training, replace any indicative values in a submission narrative with the values written to `outputs/metrics.json`.
