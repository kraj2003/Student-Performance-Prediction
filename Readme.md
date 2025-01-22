#end to end machine learning project

![image](https://github.com/user-attachments/assets/c42b3372-1c42-42f5-bfc4-7625db250a29)

# Student Performance Prediction Project

## Project Overview
The **Student Performance Prediction** project aims to predict students' academic performance based on various factors such as demographic details, past academic records, and behavioral attributes. By leveraging machine learning techniques, this project helps educators and institutions identify at-risk students and provide targeted interventions to improve outcomes.

## Features
- Data preprocessing and feature engineering for handling missing values, outliers, and encoding categorical variables.
- Model training using machine learning algorithms.
- Hyperparameter tuning for optimized model performance.
- Evaluation of models using performance metrics.
- Deployment of a RESTful API for predictions.

## Prerequisites
Ensure the following tools are installed:
- Python 3.8 or higher
- Docker

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repository/Student-Performance-Prediction.git
   cd student-performance-prediction
   ```
2. Create a virtual environment and install the required libraries:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```

## Libraries Used
- **Data Handling**:
  - `pandas`
  - `numpy`
- **Visualization**:
  - `matplotlib`
  - `seaborn`
- **Machine Learning**:
  - `scikit-learn`
  - `catboost`
  - `xgboost`
- **Utilities**:
  - `dill`
- **API Deployment**:
  - `Flask`

## Usage

### Running Locally
1. Prepare the dataset:
   - Place your dataset in the `data/` directory and ensure the filename matches the one referenced in the code.
2. Train the model:
   ```bash
   python train.py
   ```
3. Evaluate the model:
   ```bash
   python evaluate.py
   ```
4. Start the Flask server:
   ```bash
   python app.py
   ```
5. Access the API at `http://127.0.0.1:5000/`.

### Running with Docker
1. Build the Docker image:
   ```bash
   docker build -t student-performance-prediction .
   ```
2. Run the Docker container:
   ```bash
   docker run -p 5000:5000 student-performance-prediction
   ```
3. Access the API at `http://127.0.0.1:5000/`.

## Project Structure
```
student-performance-prediction/
├── app/
│   ├── main.py          # Flask application
│   └── utils.py         # Utility functions
├── data/                # Dataset files
├── models/              # Pretrained models and checkpoints
├── scripts/             # Training and evaluation scripts
├── Dockerfile           # Docker configuration
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

## Evaluation Metrics
The performance of the models is evaluated using:
- **Accuracy**: Proportion of correctly predicted instances.
- **Precision, Recall, and F1-Score**: To measure the performance on imbalanced datasets.
- **ROC-AUC Score**: For evaluating the model's ability to distinguish between classes.

## Future Improvements
- Add advanced deep learning models for predictions.
- Include a feature importance visualization module.
- Integrate a user-friendly web interface for non-technical users.

## Contributors
- [Khushi Rajpurohit](https://github.com/kraj2003)

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.
