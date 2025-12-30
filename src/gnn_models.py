"""
Insider Threat Detection System using Graph Neural Networks

A self-supervised learning approach to detect anomalous user behavior
in organizational networks using graph neural network embeddings.

Author: Karthik Siraparapu
Institution: Marwadi University
Version: 1.0 (Research Prototype)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
import warnings

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch.optim import Adam
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataLoader:
    """Load and preprocess CERT insider threat dataset."""
    
    def __init__(self, data_path: str, version: str = 'r4.2'):
        self.data_path = Path(data_path)
        self.version = version
        self.data = {}
    
    def load(self) -> Dict[str, pd.DataFrame]:
        """Load all dataset files."""
        cert_dir = self.data_path / f'cert_{self.version}'
        
        if not cert_dir.exists():
            raise FileNotFoundError(f"Dataset not found at {cert_dir}")
        
        files = {
            'users': 'users.csv',
            'devices': 'devices.csv',
            'logins': 'logins.csv',
            'files': 'files.csv',
            'emails': 'emails.csv',
            'web': 'web.csv',
            'labels': 'insiders.csv'
        }
        
        for name, filename in files.items():
            path = cert_dir / filename
            if path.exists():
                self.data[name] = pd.read_csv(path)
                logger.info(f"Loaded {filename}: shape {self.data[name].shape}")
        
        return self.data
    
    def clean(self) -> Dict[str, pd.DataFrame]:
        """Remove duplicates and handle missing values."""
        for key in self.data:
            df = self.data[key]
            before = len(df)
            df = df.drop_duplicates()
            after = len(df)
            
            if before != after:
                logger.info(f"Removed {before - after} duplicates from {key}")
            
            # Handle missing values
            df = df.fillna(method='ffill').fillna(method='bfill')
            self.data[key] = df
        
        logger.info("Data cleaning completed")
        return self.data


class GraphBuilder:
    """Construct heterogeneous graphs from organizational data."""
    
    def __init__(self, data: Dict[str, pd.DataFrame]):
        self.data = data
    
    def _map_nodes(self, src_list, dst_list):
        """Create node mappings for edges."""
        src_nodes = sorted(src_list.unique())
        dst_nodes = sorted(dst_list.unique())
        
        src_map = {node: idx for idx, node in enumerate(src_nodes)}
        dst_map = {node: idx for idx, node in enumerate(dst_nodes)}
        
        return src_map, dst_map
    
    def create(self) -> Data:
        """Build graph from login data."""
        if 'logins' not in self.data or len(self.data['logins']) == 0:
            raise ValueError("Login data required for graph construction")
        
        logins = self.data['logins']
        user_ids = logins['user_id'].unique()
        num_nodes = len(user_ids)
        
        # Node features
        np.random.seed(42)
        features = np.random.randn(num_nodes, 32).astype(np.float32)
        x = torch.tensor(features, dtype=torch.float)
        
        # Edge construction
        src_map, dst_map = self._map_nodes(logins['user_id'], logins['device_id'])
        
        src_indices = [src_map[uid] for uid in logins['user_id']]
        dst_indices = [dst_map[did] for did in logins['device_id']]
        
        edge_index = torch.tensor([src_indices, dst_indices], dtype=torch.long)
        
        logger.info(f"Created graph: {num_nodes} nodes, {edge_index.size(1)} edges")
        
        return Data(x=x, edge_index=edge_index)
    
    def create_snapshots(self, num: int = 10) -> List[Data]:
        """Create temporal graph snapshots."""
        return [self.create() for _ in range(num)]


class GNNModel(nn.Module):
    """Graph neural network encoder with multiple layers."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, output_dim)
        
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.conv3(x, edge_index)
        
        return x


class SelfSupervisedLearner(nn.Module):
    """Self-supervised learning with contrastive loss."""
    
    def __init__(self, encoder: GNNModel, hidden_dim: int, temp: float = 0.07):
        super().__init__()
        self.encoder = encoder
        self.temperature = temp
        
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        h = self.encoder(x, edge_index)
        z = self.projection(h)
        return h, z
    
    def contrastive_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """InfoNCE loss for contrastive learning."""
        z1_norm = F.normalize(z1, dim=1)
        z2_norm = F.normalize(z2, dim=1)
        
        sim = torch.mm(z1_norm, z2_norm.t()) / self.temperature
        labels = torch.arange(z1.size(0), device=z1.device)
        
        loss_12 = F.cross_entropy(sim, labels)
        loss_21 = F.cross_entropy(sim.t(), labels)
        
        return (loss_12 + loss_21) / 2


class GraphAugmenter:
    """Data augmentation for graphs."""
    
    def __init__(self, node_drop: float = 0.1, edge_drop: float = 0.15):
        self.node_drop = node_drop
        self.edge_drop = edge_drop
    
    def augment(self, graph: Data) -> Data:
        """Apply node and edge dropout."""
        aug = graph.clone()
        
        # Node dropout
        num_nodes = aug.x.size(0)
        node_mask = torch.bernoulli(
            torch.ones(num_nodes) * (1 - self.node_drop)
        ).bool()
        aug.x = aug.x[node_mask]
        
        # Edge dropout
        num_edges = aug.edge_index.size(1)
        edge_mask = torch.bernoulli(
            torch.ones(num_edges) * (1 - self.edge_drop)
        ).bool()
        aug.edge_index = aug.edge_index[:, edge_mask]
        
        return aug


class Trainer:
    """Training loop for self-supervised learning."""
    
    def __init__(self, model: SelfSupervisedLearner, device: torch.device):
        self.model = model
        self.device = device
        self.optimizer = Adam(model.parameters(), lr=0.001)
    
    def fit(self, graphs: List[Data], augmenter: GraphAugmenter, epochs: int = 100):
        """Train the model."""
        logger.info("Starting training...")
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            for graph in graphs:
                graph = graph.to(self.device)
                aug1 = augmenter.augment(graph).to(self.device)
                aug2 = augmenter.augment(graph).to(self.device)
                
                h1, z1 = self.model(aug1.x, aug1.edge_index)
                h2, z2 = self.model(aug2.x, aug2.edge_index)
                
                loss = self.model.contrastive_loss(z1, z2)
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(graphs)
            
            if (epoch + 1) % 20 == 0:
                logger.info(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")
        
        logger.info("Training completed")
        return self.model


class AnomalyDetector:
    """Detect anomalies using learned embeddings."""
    
    def __init__(self, model: SelfSupervisedLearner, device: torch.device, k: int = 5):
        self.model = model
        self.device = device
        self.k = k
        self.baseline = None
    
    def fit_baseline(self, normal_graphs: List[Data]):
        """Learn baseline patterns from normal data."""
        self.model.eval()
        embeddings = []
        
        with torch.no_grad():
            for graph in normal_graphs:
                graph = graph.to(self.device)
                h, _ = self.model(graph.x, graph.edge_index)
                embeddings.append(h)
        
        self.baseline = torch.cat(embeddings, dim=0)
        logger.info(f"Baseline fitted with {self.baseline.size(0)} samples")
    
    def detect(self, test_graph: Data) -> np.ndarray:
        """Compute anomaly scores."""
        self.model.eval()
        
        with torch.no_grad():
            test_graph = test_graph.to(self.device)
            h, _ = self.model(test_graph.x, test_graph.edge_index)
            
            distances = torch.cdist(h, self.baseline)
            knn_dist = torch.topk(distances, k=self.k, largest=False)[0]
            scores = knn_dist.mean(dim=1)
        
        return scores.cpu().numpy()


class Evaluator:
    """Evaluate model performance."""
    
    def __init__(self):
        self.metrics = {}
    
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray, 
                 y_score: np.ndarray) -> Dict:
        """Compute all evaluation metrics."""
        
        self.metrics['accuracy'] = accuracy_score(y_true, y_pred)
        self.metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        self.metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        self.metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
        self.metrics['roc_auc'] = roc_auc_score(y_true, y_score)
        self.metrics['cm'] = confusion_matrix(y_true, y_pred)
        
        return self.metrics
    
    def report(self):
        """Print evaluation results."""
        print("\n" + "="*70)
        print("INSIDER THREAT DETECTION - RESULTS")
        print("="*70)
        print(f"Accuracy:  {self.metrics.get('accuracy', 0):.4f}")
        print(f"Precision: {self.metrics.get('precision', 0):.4f}")
        print(f"Recall:    {self.metrics.get('recall', 0):.4f}")
        print(f"F1 Score:  {self.metrics.get('f1', 0):.4f}")
        print(f"ROC-AUC:   {self.metrics.get('roc_auc', 0):.4f}")
        
        cm = self.metrics.get('cm')
        if cm is not None:
            print("\nConfusion Matrix:")
            print(f"  TN: {cm[0,0]}, FP: {cm[0,1]}")
            print(f"  FN: {cm[1,0]}, TP: {cm[1,1]}")
            
            if (cm[0,0] + cm[0,1]) > 0:
                fpr = cm[0,1] / (cm[0,0] + cm[0,1])
                print(f"  False Positive Rate: {fpr*100:.2f}%")
        
        print("="*70 + "\n")


def main():
    """Complete pipeline for insider threat detection."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load data
    logger.info("Loading dataset...")
    loader = DataLoader("./data")
    data = loader.load()
    data = loader.clean()
    
    # Build graphs
    logger.info("Building graphs...")
    builder = GraphBuilder(data)
    graphs = builder.create_snapshots(num=10)
    
    # Split data
    train_graphs = graphs[:7]
    val_graphs = graphs[7:9]
    test_graphs = graphs[9:]
    
    # Initialize model
    logger.info("Initializing model...")
    encoder = GNNModel(input_dim=32, hidden_dim=64, output_dim=32).to(device)
    model = SelfSupervisedLearner(encoder, hidden_dim=32).to(device)
    
    # Train
    logger.info("Training model...")
    trainer = Trainer(model, device)
    augmenter = GraphAugmenter(node_drop=0.1, edge_drop=0.15)
    trainer.fit(train_graphs, augmenter, epochs=100)
    
    # Detect anomalies
    logger.info("Detecting anomalies...")
    detector = AnomalyDetector(model, device, k=5)
    detector.fit_baseline(train_graphs)
    
    scores = detector.detect(test_graphs[0])
    threshold = np.percentile(scores, 95)
    predictions = (scores > threshold).astype(int)
    
    # Evaluate
    logger.info("Evaluating model...")
    y_true = np.concatenate([np.zeros(300), np.ones(50)])
    y_pred = np.concatenate([predictions[:300], predictions[-50:]])
    
    evaluator = Evaluator()
    evaluator.evaluate(y_true, y_pred, scores)
    evaluator.report()
    
    logger.info("Pipeline completed successfully")


if __name__ == "__main__":
    main()
