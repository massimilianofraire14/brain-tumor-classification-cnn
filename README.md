# Brain Tumor Classification using CNN

Fraire Massimiliano

## Abstract

This project explores the potential of the VGG16 convolutional neural network (CNN) architecture, using transfer learning, for brain tumor classification based on MRI images. We fine-tune the pre-trained model and employ visualization techniques such as saliency maps and Grad-CAM to interpret predictions. A comparison is made between a VGG16-based model and a custom 3-layer CNN to assess the benefits of transfer learning.

## Task

We apply a deep learning approach, leveraging a pre-trained VGG16 model, to classify MRI images into four categories without requiring prior tumor segmentation. To better understand the model’s decisions, visualization techniques like Grad-CAM and saliency maps are used.

## Methods

### Dataset
- **Source**: 7023 human brain MRI images categorized into Glioma, Meningioma, No Tumor, and Pituitary classes.
- **Download**: [Brain Tumor MRI Dataset by Masoud Nickparvar](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- **Preprocessing**: 
  - Duplicate removal with MD5 hashing
  - Resized to 224x224x3
  - Normalization to [0,1] range
  - Stratified 80/10/10 train-validation-test split
- **Data Augmentation**:
  - Random rotations, brightness adjustments, shifts, distortions, and flips

### Baseline Model
- **Softmax Logistic Regression**: Achieved 88% accuracy.

### Models
- **Custom 3-Layer CNN**: 
  - Three convolutional layers with 32, 64, and 128 filters
  - 2x2 max pooling after each convolution
  - Dense layer with 128 neurons
  - L2 regularization (0.001)

- **Modified VGG16**:
  - Pre-trained VGG16 with first 4 blocks frozen
  - Custom top layers with Global Average Pooling, Dropout (0.05), Dense layer (128 units, L2 regularization 0.1), and a Softmax output

### Training
- **Loss Function**: Categorical Cross-Entropy
- **Optimizer**: Rectified Adam (RAdam)
- **Learning Rate Scheduler**: ReduceLROnPlateau

### Visualization Techniques
- **Saliency Maps**: Highlights regions influencing predictions.
- **Grad-CAM**: Visualizes important regions from the final convolutional layers.

## Results

- **Custom 3-layer CNN**: 
  - 88% accuracy with data augmentation
  - Lower F1 score for "Meningioma" class

- **VGG16-based Model**:
  - 97% accuracy on the test set
  - Highest precision, recall, and F1 scores, especially improved tumor localization

- **Visualization**:
  - VGG16 model shows localized tumor regions.
  - Custom CNN displays less focused attention on relevant areas.

## Conclusion

The VGG16-based transfer learning model significantly outperformed the custom 3-layer CNN, achieving 97% test accuracy. Visualizations confirmed better feature extraction by VGG16, highlighting its suitability for brain tumor diagnosis directly from MRI images without prior segmentation.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/brain-tumor-classification-cnn.git
cd brain-tumor-classification-cnn
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## How to Run

1. Place the dataset as described in the Methods section.
2. Run the Jupyter Notebook:
```bash
jupyter notebook Brain_Tumor_Classification_CNN.ipynb
```

## License

This project is licensed under the MIT License.

---

> References:
> 1. Nickparvar, M. (2024). Brain Tumor MRI Dataset. Kaggle.
> 2. Khaliki, M. Z., & Başarslan, M. S. (2024). Nature Scientific Reports.
> 3. Chmiel, W., et al. (2023). Sensors.
> 4. Liu, L., et al. (1970). University of Illinois Urbana-Champaign.


