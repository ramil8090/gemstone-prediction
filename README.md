# Gemstone Prediction
A capstone1 project for the Machine Learning Zoomcamp 2025

## 📁 Repository Structure  
- `notebook.ipynb`: Data exploration, model training, and hyperparameter tuning.
- `train.ipynb`: Notebook to train the final model and export it to `.onnx`.
- `lambda_function.py`: AWS Lambda handler for serving predictions.
- `test.py`: Script to verify the running Docker/Lambda service.
- `Dockerfile`: Instructions for building the containerized environment.

✅ Why This Project Matters
Identifying gemstones manually requires years of training and specialized equipment like refractometers. This project demonstrates how Computer Vision can:
- **Democratize Expertise**: Allow non-experts to get an initial identification of a stone using just a smartphone camera.
- **Scale Identification**: Process thousands of images for inventory management in jewelry e-commerce.
- **High-Stakes Accuracy**: Leverage Deep Learning to distinguish between stones that look visually similar to the naked eye but have distinct crystalline structures.

## 📊 Results & Findings
During the model selection phase, three major architectures were evaluated to find the best balance between depth and feature extraction:
- **MobileNet V2**: Fast and lightweight, but struggled with the fine-grained details of similar gemstones.
- **EfficientNet B4**: Provided good feature scaling but was prone to overfitting on the specific dataset size.
- **ResNet101** *Selected Model*: The deep residual layers allowed for better extraction of complex facet patterns and color gradients.

### Key Insights:
- **Final Accuracy**: The model achieved a validation accuracy of approximately 70%.
- **Observations**: The 70% accuracy reflects the high complexity of the 87-class dataset. Many gemstones (e.g., different varieties of Quartz or Garnet) are visually nearly identical in standard photography, suggesting that future iterations might benefit from higher-resolution inputs or specialized lighting data.

## 🛠 Quick Start / Usage  
### Prerequisites  
- Python 3.x  
- Google Colab account (recommended for GPU training)
- Docker (for deployment)

### Train model:
- Open notebooks/notebook.ipynb in Google Colab
- Switch to a T4 GPU processor
- Run all cells
- Download the trained ONNX model along with onnx.data file to the ROOT of the current repository

### Running Prediction Lambda using Docker
The application is containerized to run as an AWS Lambda function using the Lambda Runtime Interface Client (RIC).
1. Build the image:
```bash
docker build -t gemstone-prediction .
```
2. Run the container:
```bash
docker run -it --rm -p 8080:8080 gemstone-prediction:latest
```

### Running Tests
With the Docker container running, execute the test script to send a sample image URL to the endpoint:
```bash
python test.py
```

## 📚 Acknowledgements & References
- Dataset from Kaggle: “Gemstones Images” by Daria Chemkaeva (https://www.kaggle.com/datasets/lsind18/gemstones-images/data).
- Inspiration: Inspired by industrial gemstone grading standards and automated jewelry cataloging.
- Built as part of the Machine Learning Zoomcamp 2025 capstone1 assignment.
