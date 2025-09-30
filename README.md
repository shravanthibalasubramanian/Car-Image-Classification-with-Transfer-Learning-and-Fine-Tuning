# Car-Image-Classification-with-Transfer-Learning-and-Fine-Tuning
Through deep learning techniques, my project performs image classification using pretrained CNNs — ResNet50V2, InceptionV3, and DenseNet121. Both transfer learning (TL) and fine-tuning (FT) approaches are implemented and compared on the Car Images Dataset from Kaggle.

## Dataset
- **Dataset Used**: Car Image Dataset  
- **Train/Test Split**: 75% training / 25% testing  
- **Classes**:  
  - Audi  
  - Hyundai Creta  
  - Mahindra Scorpio  
  - Rolls Royce  
  - Swift  
  - Tata Safari  
  - Toyota Innova  
- **Total Images**: 1,045 (after split)  

---

## Models & Training

### Transfer Learning (TL)
- ResNet50V2 → BatchNorm + Dropout (25%) before output  
- InceptionV3 → BatchNorm + Dropout (35%) before output  
- DenseNet121 → Dropout (15%) before output  
- Each model trained for **10 epochs** with **10% validation split**.  

### Fine-Tuning (FT)
- ResNet50V2 → first 25% layers frozen  
- InceptionV3 → first 35% layers frozen  
- DenseNet121 → all layers trainable  
- Each model fine-tuned for **10 epochs**.  

---

## Results Summary

| Model        | Training Type     | Accuracy | Precision | Recall | F1-Score |
|--------------|-----------------|----------|-----------|--------|----------|
| ResNet50V2   | Transfer Learning | 0.88     | 0.88      | 0.86   | 0.87     |
| ResNet50V2   | Fine Tuning       | 0.93     | 0.92      | 0.93   | 0.92     |
| InceptionV3  | Transfer Learning | 0.86     | 0.84      | 0.82   | 0.83     |
| InceptionV3  | Fine Tuning       | 0.95     | 0.95      | 0.94   | 0.95     |
| DenseNet121  | Transfer Learning | 0.89     | 0.89      | 0.88   | 0.88     |
| DenseNet121  | Fine Tuning       | 0.98     | 0.98      | 0.98   | 0.98     |

**Note:** DenseNet121 with Fine-Tuning achieved the best performance (98% accuracy).  

---

## Evaluation Metrics
Each model was evaluated using:
- **Confusion Matrix** (saved in `results/`)  
- **Precision, Recall, F1-Score** (macro averages used)  
- **metrics_summary.csv** file includes a summary of all models  

---

## Repository Structure
```
DeepLearning-ImageClassification/
├── notebooks/                 
│   ├── Model1_TL.ipynb
│   ├── Model1_FT.ipynb
│   ├── Model2_TL.ipynb
│   ├── Model2_FT.ipynb
│   ├── Model3_TL.ipynb
│   └── Model3_FT.ipynb
├── results/                  
│   ├── confusion_matrix_resnet.png
│   ├── confusion_matrix_inception.png
│   ├── confusion_matrix_densenet.png
│   └── metrics_summary.csv
├── README.md                
├── requirements.txt        
└── .gitignore
```
---

## Installation & Setup
1. Clone the repository:

git clone https://github.com/shravanthibalasubramanian/Car-Image-Classification-with-Transfer-Learning-and-Fine-Tuning.git
cd Car-Image-Classification-with-Transfer-Learning-and-Fine-Tuning

## Install dependencies:
pip install -r requirements.txt
Open any notebook in notebooks/ using Jupyter Notebook or Google Colab.
Run all cells to train or evaluate models.

## Usage
Training: Run notebooks in notebooks/ folder to train TL/FT models.
Evaluation: Metrics and confusion matrices are saved in results/.
Visualization: Plot confusion matrices for comparison.

## Tech Stack
Python 3.x
TensorFlow / Keras
scikit-learn
Matplotlib / Seaborn
Google Colab (GPU runtime recommended)

## Author
Shravanthi Balasubramanian
Email: shravanthi.balasubramanian@gmail.com
