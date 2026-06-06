"""
Data Loading and Preprocessing Framework
========================================

This module provides comprehensive utilities for loading, preprocessing, and managing
medical datasets for causal discovery experiments. The framework supports multiple
clinical datasets (Diabetes, Heart Disease, Hepatitis) with robust preprocessing
pipelines including missing value handling, feature encoding, normalization, and
data quality validation.

The implementation ensures consistent data preparation across experimental conditions,
enabling fair algorithmic comparisons and reproducible research. The framework also
provides utilities for creating synthetic sample datasets and managing data storage
structures.

Author: Anonymous (for review)
Date: 2025
"""

import pandas as pd
import numpy as np
import os
import logging
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import requests
import zipfile
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MedicalDataLoader:
    """
    A class to load and preprocess medical datasets for causal discovery.
    """
    
    def __init__(self, data_dir: str = "data/raw"):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Directory containing the datasets
        """
        # Support both data/raw and data/Raw (user provided)
        preferred_dirs = [Path(data_dir), Path("data/Raw"), Path("data/raw")]
        existing = [p for p in preferred_dirs if p.exists()]
        self.data_dir = existing[0] if existing else preferred_dirs[0]
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Dataset configurations
        self.dataset_configs = {
            'heart_disease': {
                'filename': 'heart.csv',
                'url': 'https://www.kaggle.com/datasets/ronitf/heart-disease-uci/download',
                'target_column': 'target',
                'categorical_columns': ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'],
                'numerical_columns': ['age', 'trestbps', 'chol', 'thalach', 'oldpeak'],
                'description': 'Heart Disease UCI Dataset'
            },
            'diabetes': {
                'filename': 'diabetes.csv',
                'url': 'https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database/download',
                'target_column': 'Outcome',
                'categorical_columns': [],
                'numerical_columns': ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                                    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'],
                'description': 'Pima Indians Diabetes Database'
            },
            'hepatitis': {
                'filename': 'hepatitis.csv',
                'url': 'https://www.kaggle.com/datasets/fedesoriano/hepatitis-c-dataset/download',
                'target_column': 'class',  # Actual column name in data file
                'categorical_columns': ['sex', 'class', 'steroid', 'antivirals', 'fatigue', 'malaise', 'anorexia', 
                                        'liver_big', 'liver_firm', 'spleen_palable', 'spiders', 'ascites', 'varices', 'histology'],
                'numerical_columns': ['age', 'bilirubin', 'alk_phosphate', 'sgot', 'albumin', 'protime'],
                'description': 'Hepatitis C Dataset'
            }
        }
    
    def download_dataset(self, dataset_name: str) -> bool:
        """
        Download dataset from Kaggle (requires kaggle API setup).
        
        Args:
            dataset_name: Name of the dataset to download
            
        Returns:
            True if successful, False otherwise
        """
        if dataset_name not in self.dataset_configs:
            logger.error(f"Unknown dataset: {dataset_name}")
            return False
        
        config = self.dataset_configs[dataset_name]
        file_path = self.data_dir / config['filename']
        
        if file_path.exists():
            logger.info(f"Dataset {dataset_name} already exists")
            return True
        
        try:
            # Note: This requires kaggle API setup
            # For now, we'll create a placeholder function
            logger.warning(f"Please manually download {dataset_name} from {config['url']}")
            logger.warning(f"Save it as {file_path}")
            return False
            
        except Exception as e:
            logger.error(f"Failed to download {dataset_name}: {e}")
            return False
    
    def _find_file(self, expected_filename: str, keywords: List[str]) -> Optional[Path]:
        """
        Find a dataset file by exact name or by keyword match in data directory.
        """
        exact = self.data_dir / expected_filename
        if exact.exists():
            return exact
        # search common extensions
        candidates = list(self.data_dir.glob("*.csv")) + list(self.data_dir.glob("*.xlsx")) + list(self.data_dir.glob("*.xls"))
        for p in candidates:
            name = p.name.lower()
            if any(k in name for k in keywords):
                return p
        return None

    def load_dataset(self, dataset_name: str) -> pd.DataFrame:
        """
        Load a medical dataset.
        
        Args:
            dataset_name: Name of the dataset to load
            
        Returns:
            Loaded DataFrame
        """
        if dataset_name not in self.dataset_configs:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        config = self.dataset_configs[dataset_name]
        # Try to locate file robustly
        keyword_map = {
            'heart_disease': ['heart', 'cardio'],
            'diabetes': ['diab'],
            'hepatitis': ['hepat']
        }
        file_path = self._find_file(config['filename'], keyword_map.get(dataset_name, []))
        
        if file_path is None or not file_path.exists():
            logger.warning(f"Dataset file not found: {file_path}")
            logger.info("Attempting to download...")
            if not self.download_dataset(dataset_name):
                raise FileNotFoundError(f"Dataset {dataset_name} not found and download failed")
            file_path = self.data_dir / config['filename']

        # Load the dataset
        try:
            if str(file_path).lower().endswith('.csv'):
                data = pd.read_csv(file_path)
            else:
                data = pd.read_excel(file_path)
            logger.info(f"Loaded {dataset_name}: {data.shape}")
            return data
        except Exception as e:
            logger.error(f"Failed to load {dataset_name}: {e}")
            raise
    
    def preprocess_dataset(self, data: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """
        Preprocess a medical dataset for causal discovery.
        
        Args:
            data: Raw dataset
            dataset_name: Name of the dataset
            
        Returns:
            Preprocessed DataFrame
        """
        if dataset_name not in self.dataset_configs:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        config = self.dataset_configs[dataset_name]
        processed_data = data.copy()

        # Drop spurious row-index / ID columns that are not clinical variables
        # (e.g. the 'no' column in Heart.csv). These are not causal variables and
        # must not enter the causal graph. Match common index-column names.
        index_like = {"no", "id", "index", "unnamed: 0", "row", "patient_id", "patientid"}
        drop_cols = [c for c in processed_data.columns if str(c).strip().lower() in index_like]
        if drop_cols:
            processed_data = processed_data.drop(columns=drop_cols)
            logger.info(f"Dropped non-clinical index column(s): {drop_cols}")

        # Infer target column if missing
        target = config['target_column']
        if target not in processed_data.columns:
            possible_targets = ['target', 'outcome', 'class', 'label', 'category']
            for t in possible_targets:
                for col in processed_data.columns:
                    if col.lower() == t:
                        target = col
                        break
            config['target_column'] = target

        # Handle missing values
        logger.info("Handling missing values...")
        for col in processed_data.columns:
            if processed_data[col].dtype == 'object':
                # Categorical columns: fill with the mode.
                mode_value = processed_data[col].mode()
                if len(mode_value) > 0:
                    processed_data[col] = processed_data[col].fillna(mode_value[0])
            else:
                # Numerical columns: fill with the median.
                median_value = processed_data[col].median()
                processed_data[col] = processed_data[col].fillna(median_value)
        
        # Encode categorical variables
        logger.info("Encoding categorical variables...")
        label_encoders = {}
        for col in config['categorical_columns']:
            if col in processed_data.columns:
                le = LabelEncoder()
                processed_data[col] = le.fit_transform(processed_data[col].astype(str))
                label_encoders[col] = le
        
        # Standardize numerical variables
        logger.info("Standardizing numerical variables...")
        scaler = StandardScaler()
        numerical_cols = [col for col in config['numerical_columns'] if col in processed_data.columns]
        if numerical_cols:
            processed_data[numerical_cols] = scaler.fit_transform(processed_data[numerical_cols])
        
        # Remove any remaining missing values
        processed_data = processed_data.dropna()
        
        logger.info(f"Preprocessed {dataset_name}: {processed_data.shape}")
        return processed_data
    
    def get_effect_variables(self, dataset_name: str) -> List[str]:
        """
        Get the effect variables for a dataset based on domain knowledge.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            List of effect variable names
        """
        # Resolve target/effect variable dynamically
        config = self.dataset_configs.get(dataset_name, {})
        candidate = config.get('target_column', None)
        if candidate:
            return [candidate]
        
        # Fallback: try to load data and find effect variable
        try:
            data = self.load_dataset(dataset_name)
            # Check for common effect variable names
            for name in ['class', 'target', 'Outcome', 'Category', 'outcome', 'category']:
                if name in data.columns:
                    return [name]
        except:
            pass
        
        # Final fallback common names
        fallback_names = {
            'heart_disease': 'target',
            'diabetes': 'Outcome',
            'hepatitis': 'class'
        }
        if dataset_name in fallback_names:
            return [fallback_names[dataset_name]]
        
        return []
    
    def get_causal_relationships(self, dataset_name: str) -> Dict[str, List[str]]:
        """
        Get known causal relationships for each dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Dictionary mapping causes to effects
        """
        causal_relationships = {
            'heart_disease': {
                'age': ['target'],
                'sex': ['target'],
                'cp': ['target'],
                'trestbps': ['target'],
                'chol': ['target'],
                'fbs': ['target'],
                'restecg': ['target'],
                'thalach': ['target'],
                'exang': ['target'],
                'oldpeak': ['target'],
                'slope': ['target'],
                'ca': ['target'],
                'thal': ['target']
            },
            'diabetes': {
                'Pregnancies': ['Outcome'],
                'Glucose': ['Outcome'],
                'BloodPressure': ['Outcome'],
                'SkinThickness': ['Outcome'],
                'Insulin': ['Outcome'],
                'BMI': ['Outcome'],
                'DiabetesPedigreeFunction': ['Outcome'],
                'Age': ['Outcome']
            },
            'hepatitis': {
                'Age': ['Category'],
                'Sex': ['Category'],
                'ALB': ['Category'],
                'ALP': ['Category'],
                'ALT': ['Category'],
                'AST': ['Category'],
                'BIL': ['Category'],
                'CHE': ['Category'],
                'CHOL': ['Category'],
                'CREA': ['Category'],
                'GGT': ['Category'],
                'PROT': ['Category']
            }
        }
        
        return causal_relationships.get(dataset_name, {})
    
    def load_all_datasets(self) -> Dict[str, pd.DataFrame]:
        """
        Load and preprocess all medical datasets.
        
        Returns:
            Dictionary mapping dataset names to preprocessed DataFrames
        """
        datasets = {}
        
        for dataset_name in self.dataset_configs.keys():
            try:
                logger.info(f"Loading and preprocessing {dataset_name}...")
                raw_data = self.load_dataset(dataset_name)
                processed_data = self.preprocess_dataset(raw_data, dataset_name)
                datasets[dataset_name] = processed_data
                logger.info(f"Successfully processed {dataset_name}")
            except Exception as e:
                logger.error(f"Failed to process {dataset_name}: {e}")
                continue
        
        return datasets
    
    def save_processed_dataset(self, data: pd.DataFrame, dataset_name: str, 
                             output_dir: str = "data/processed") -> str:
        """
        Save processed dataset to file.
        
        Args:
            data: Processed DataFrame
            dataset_name: Name of the dataset
            output_dir: Output directory
            
        Returns:
            Path to saved file
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        file_path = output_path / f"{dataset_name}_processed.csv"
        data.to_csv(file_path, index=False)
        
        logger.info(f"Saved processed dataset to {file_path}")
        return str(file_path)
    
    def get_dataset_info(self, dataset_name: str) -> Dict:
        """
        Get information about a dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Dictionary with dataset information
        """
        if dataset_name not in self.dataset_configs:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        config = self.dataset_configs[dataset_name]
        
        try:
            data = self.load_dataset(dataset_name)
            processed_data = self.preprocess_dataset(data, dataset_name)
            
            info = {
                'name': dataset_name,
                'description': config['description'],
                'original_shape': data.shape,
                'processed_shape': processed_data.shape,
                'target_column': config['target_column'],
                'categorical_columns': config['categorical_columns'],
                'numerical_columns': config['numerical_columns'],
                'effect_variables': self.get_effect_variables(dataset_name),
                'causal_relationships': self.get_causal_relationships(dataset_name),
                'missing_values_original': data.isnull().sum().sum(),
                'missing_values_processed': processed_data.isnull().sum().sum()
            }
            
            return info
            
        except Exception as e:
            logger.error(f"Failed to get info for {dataset_name}: {e}")
            return {'name': dataset_name, 'error': str(e)}


def create_sample_datasets() -> Dict[str, pd.DataFrame]:
    """
    Create sample datasets for testing when real datasets are not available.
    
    Returns:
        Dictionary of sample datasets
    """
    np.random.seed(42)
    
    # Heart Disease Sample
    n_samples = 1000
    heart_data = pd.DataFrame({
        'age': np.random.normal(55, 10, n_samples),
        'sex': np.random.choice([0, 1], n_samples),
        'cp': np.random.choice([0, 1, 2, 3], n_samples),
        'trestbps': np.random.normal(130, 20, n_samples),
        'chol': np.random.normal(250, 50, n_samples),
        'fbs': np.random.choice([0, 1], n_samples),
        'restecg': np.random.choice([0, 1, 2], n_samples),
        'thalach': np.random.normal(150, 20, n_samples),
        'exang': np.random.choice([0, 1], n_samples),
        'oldpeak': np.random.exponential(1, n_samples),
        'slope': np.random.choice([0, 1, 2], n_samples),
        'ca': np.random.choice([0, 1, 2, 3], n_samples),
        'thal': np.random.choice([0, 1, 2], n_samples),
        'target': np.random.choice([0, 1], n_samples)
    })
    
    # Add some causal relationships
    heart_data['target'] = (
        0.3 * (heart_data['age'] > 60).astype(int) +
        0.2 * heart_data['sex'] +
        0.2 * (heart_data['chol'] > 250).astype(int) +
        0.1 * heart_data['cp'] +
        np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    ).clip(0, 1)
    
    # Diabetes Sample
    diabetes_data = pd.DataFrame({
        'Pregnancies': np.random.poisson(3, n_samples),
        'Glucose': np.random.normal(120, 30, n_samples),
        'BloodPressure': np.random.normal(70, 15, n_samples),
        'SkinThickness': np.random.normal(20, 10, n_samples),
        'Insulin': np.random.exponential(80, n_samples),
        'BMI': np.random.normal(32, 7, n_samples),
        'DiabetesPedigreeFunction': np.random.exponential(0.5, n_samples),
        'Age': np.random.normal(33, 12, n_samples),
        'Outcome': np.random.choice([0, 1], n_samples)
    })
    
    # Add causal relationships
    diabetes_data['Outcome'] = (
        0.4 * (diabetes_data['Glucose'] > 140).astype(int) +
        0.3 * (diabetes_data['BMI'] > 30).astype(int) +
        0.2 * (diabetes_data['Age'] > 40).astype(int) +
        np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
    ).clip(0, 1)
    
    # Hepatitis Sample
    hepatitis_data = pd.DataFrame({
        'Age': np.random.normal(45, 15, n_samples),
        'Sex': np.random.choice([0, 1], n_samples),
        'ALB': np.random.normal(4.5, 0.5, n_samples),
        'ALP': np.random.exponential(80, n_samples),
        'ALT': np.random.exponential(30, n_samples),
        'AST': np.random.exponential(25, n_samples),
        'BIL': np.random.exponential(0.8, n_samples),
        'CHE': np.random.normal(7, 2, n_samples),
        'CHOL': np.random.normal(4.5, 1, n_samples),
        'CREA': np.random.normal(0.9, 0.2, n_samples),
        'GGT': np.random.exponential(30, n_samples),
        'PROT': np.random.normal(7, 0.5, n_samples),
        'Category': np.random.choice([0, 1, 2], n_samples)
    })
    
    # Add causal relationships
    hepatitis_data['Category'] = (
        0.3 * (hepatitis_data['ALT'] > 40).astype(int) +
        0.2 * (hepatitis_data['AST'] > 35).astype(int) +
        0.2 * (hepatitis_data['BIL'] > 1.2).astype(int) +
        np.random.choice([0, 1, 2], n_samples, p=[0.6, 0.3, 0.1])
    )
    
    return {
        'heart_disease': heart_data,
        'diabetes': diabetes_data,
        'hepatitis': hepatitis_data
    }


def load_medical_datasets() -> Dict[str, pd.DataFrame]:
    """
    Convenience function to load all medical datasets.
    
    Returns:
        Dictionary mapping dataset names to preprocessed DataFrames
    """
    loader = MedicalDataLoader()
    
    # First try to load real datasets
    try:
        datasets = loader.load_all_datasets()
        if datasets:
            return datasets
    except Exception as e:
        logger.warning(f"Failed to load real datasets: {e}")
    
    # If real datasets fail, create sample datasets
    logger.info("Creating sample datasets for testing...")
    return create_sample_datasets()


if __name__ == "__main__":
    # Example usage
    loader = MedicalDataLoader()
    
    # Create sample datasets for testing
    sample_datasets = create_sample_datasets()
    
    # Save sample datasets
    for name, data in sample_datasets.items():
        file_path = loader.data_dir / f"{name}.csv"
        data.to_csv(file_path, index=False)
        print(f"Created sample dataset: {file_path}")
    
    # Load and preprocess
    datasets = loader.load_all_datasets()
    
    for name, data in datasets.items():
        print(f"\n{name}:")
        print(f"Shape: {data.shape}")
        print(f"Columns: {list(data.columns)}")
        print(f"Effect variables: {loader.get_effect_variables(name)}")
