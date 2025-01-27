# prj_Graphtransformers
# GraphTransformers Project

This project uses Graph Transformers and Graph Neural Networks (GNNs) to analyze the Open University Learning Analytics Dataset (OULAD). The goal is to compare different learning methods for handling relational and temporal data in the context of online education. The project includes data preprocessing, model training, and evaluation steps to examine the performance of GNN and Graph Transformer architectures on student data.

## Table of Contents

1. [Project Structure](#project-structure)
2. [Installation](#installation)
3. [Data Processing](#data-processing)
4. [Model Training and Evaluation](#model-training-and-evaluation)
5. [Notebooks for Data Exploration](#notebooks-for-data-exploration)
6. [Dependencies](#dependencies)
7. [Contributing](#contributing)

## Project Structure

The project is organized as follows:

- **data**: Directory for storing data files.
  - **raw**: Contains original data files, including:
    - `assessments.csv`
    - `courses.csv`
    - `studentAssessment.csv`
    - `studentInfo.csv`
    - `studentRegistration.csv`
    - `studentVle.csv`
    - `vle.csv`
  - **processed**: Stores processed datasets after cleaning and transformation.

- **myenv**: Virtual environment directory for project dependencies.

- **notebooks**: Jupyter notebooks for data exploration and analysis.
  - `eda.ipynb`: Exploratory Data Analysis notebook to visualize and understand the dataset.

- **scripts**: Contains shell scripts for data download and training automation.
  - `download_data.sh`: Downloads datasets if they are hosted externally.
  - `run_training.sh`: Automates the training process.

- **src**: Source code directory containing the main modules.
  - `data_processing.py`: Loads and preprocesses the data.
  - `evaluate.py`: Evaluates model performance using accuracy, F1 score, and confusion matrix.
  - `model.py`: Defines the GNN and Graph Transformer model architectures.
  - `train.py`: Script for training the models.

- **requirements.txt**: Lists all project dependencies.
- **README.md**: Documentation file describing the project.

## Installation

To set up this project locally, follow these steps:

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd prj_Graphtransformers
   ```

2. **Set Up Virtual Environment**
   ```bash
   python -m venv myenv
   source myenv/bin/activate  # On Windows, use: myenv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Data Processing

To prepare the data for model training, use the `data_processing.py` script. This script loads the raw data, processes it, and creates a graph structure suitable for GNN and Graph Transformer models.

1. **Download Data** (if hosted externally)
   Run the `download_data.sh` script to fetch the data (modify the script if necessary to point to the correct data source):

   ```bash
   bash scripts/download_data.sh
   ```

2. **Process Data**
   Run the following command to process the data:

   ```bash
   python src/data_processing.py
   ```

   This will load data from `data/raw`, process it into a graph structure, and prepare it for model training.

## Model Training and Evaluation

1. **Training the Model**
   Run the `train.py` script to train both the GNN and Graph Transformer models:

   ```bash
   python src/train.py
   ```

2. **Evaluating the Model**
   After training, evaluate the model performance by running the `evaluate.py` script:

   ```bash
   python src/evaluate.py
   ```

   The script will output the accuracy, F1 score, and confusion matrix for both models, helping to compare their performance.

## Notebooks for Data Exploration

Use the Jupyter notebook in the `notebooks` directory for data exploration and visualization:

```bash
jupyter notebook notebooks/eda.ipynb
```

This notebook contains visualizations and exploratory data analysis to better understand the dataset.

## Dependencies

Ensure your `requirements.txt` file includes the following dependencies:

```text
torch
torchvision
dgl
torch_geometric
scikit-learn
pandas
networkx
jupyter
```

To install these dependencies, run:

```bash
pip install -r requirements.txt
```

