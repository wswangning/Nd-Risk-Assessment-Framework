"""
AIVIVE (AI-enhanced In Vitro to In Vivo Extrapolation) model using conditional GAN.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class Generator(nn.Module):
    """Generator network for cGAN-based AIVIVE model."""
    
    def __init__(self, input_dim: int, latent_dim: int = 100, 
                 hidden_dims: List[int] = [512, 256]):
        """
        Initialize Generator.
        
        Parameters
        ----------
        input_dim : int
            Input dimension (number of genes)
        latent_dim : int, optional
            Dimension of latent space (default: 100)
        hidden_dims : List[int], optional
            Hidden layer dimensions (default: [512, 256])
        """
        super(Generator, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Build generator layers
        layers = []
        current_dim = input_dim + latent_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            current_dim = hidden_dim
            
        # Output layer
        layers.append(nn.Linear(current_dim, input_dim))
        layers.append(nn.Tanh())  # Output in [-1, 1] range
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            In vitro transcriptomic data
        z : torch.Tensor
            Noise vector
            
        Returns
        -------
        torch.Tensor
            Predicted in vivo transcriptomic data
        """
        # Concatenate input and noise
        x_combined = torch.cat([x, z], dim=1)
        return self.model(x_combined)


class Discriminator(nn.Module):
    """Discriminator network for cGAN-based AIVIVE model."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [512, 256]):
        """
        Initialize Discriminator.
        
        Parameters
        ----------
        input_dim : int
            Input dimension (number of genes)
        hidden_dims : List[int], optional
            Hidden layer dimensions (default: [512, 256])
        """
        super(Discriminator, self).__init__()
        
        # Input is concatenation of real/fake and condition
        layers = []
        current_dim = input_dim * 2  # Real/Fake + Condition
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.Dropout(0.3))
            current_dim = hidden_dim
            
        # Output layer
        layers.append(nn.Linear(current_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Real or generated transcriptomic data
        condition : torch.Tensor
            Condition (in vitro data)
            
        Returns
        -------
        torch.Tensor
            Probability that input is real
        """
        x_combined = torch.cat([x, condition], dim=1)
        return self.model(x_combined)


class LocalOptimizer(nn.Module):
    """Local optimizer for key toxicity pathways."""
    
    def __init__(self, input_dim: int, key_gene_indices: List[int], 
                 hidden_dim: int = 128):
        """
        Initialize LocalOptimizer.
        
        Parameters
        ----------
        input_dim : int
            Input dimension
        key_gene_indices : List[int]
            Indices of key toxicity pathway genes
        hidden_dim : int, optional
            Hidden layer dimension (default: 128)
        """
        super(LocalOptimizer, self).__init__()
        
        self.key_gene_indices = key_gene_indices
        self.num_key_genes = len(key_gene_indices)
        
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, self.num_key_genes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Transcriptomic data
            
        Returns
        -------
        torch.Tensor
            Optimized predictions for key genes
        """
        return self.model(x)


class AIVIVEModel:
    """Main AIVIVE model integrating all components."""
    
    def __init__(self, config: Dict, device: str = 'cpu'):
        """
        Initialize AIVIVE model.
        
        Parameters
        ----------
        config : Dict
            Model configuration
        device : str, optional
            Device for training (default: 'cpu')
        """
        self.config = config
        self.device = torch.device(device)
        
        # Model dimensions
        self.input_dim = config.get('input_dim', 15000)
        self.latent_dim = config.get('latent_dim', 100)
        self.key_gene_indices = config.get('key_gene_indices', list(range(347)))
        
        # Initialize components
        self.generator = Generator(self.input_dim, self.latent_dim).to(self.device)
        self.discriminator = Discriminator(self.input_dim).to(self.device)
        self.local_optimizer = LocalOptimizer(
            self.input_dim, self.key_gene_indices
        ).to(self.device)
        
        # Optimizers
        self.g_optimizer = optim.Adam(
            self.generator.parameters(),
            lr=config.get('learning_rate', 0.0002),
            betas=(0.5, 0.999)
        )
        self.d_optimizer = optim.Adam(
            self.discriminator.parameters(),
            lr=config.get('learning_rate', 0.0002),
            betas=(0.5, 0.999)
        )
        self.l_optimizer = optim.Adam(
            self.local_optimizer.parameters(),
            lr=config.get('learning_rate', 0.0001)
        )
        
        # Loss functions
        self.criterion = nn.BCELoss()
        self.mse_loss = nn.MSELoss()
        
        # Training history
        self.history = {
            'g_loss': [], 'd_loss': [], 'l_loss': [],
            'cosine_similarity': [], 'pathway_accuracy': []
        }
        
    def train(self, train_data: Dict, val_data: Optional[Dict] = None, 
              epochs: int = 100, batch_size: int = 32):
        """
        Train the AIVIVE model.
        
        Parameters
        ----------
        train_data : Dict
            Training data with keys: 'in_vitro', 'in_vivo'
        val_data : Dict, optional
            Validation data
        epochs : int, optional
            Number of training epochs (default: 100)
        batch_size : int, optional
            Batch size (default: 32)
        """
        logger.info(f"Starting AIVIVE model training for {epochs} epochs")
        
        # Prepare data
        in_vitro_train = torch.FloatTensor(train_data['in_vitro']).to(self.device)
        in_vivo_train = torch.FloatTensor(train_data['in_vivo']).to(self.device)
        
        dataset = torch.utils.data.TensorDataset(in_vitro_train, in_vivo_train)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )
        
        for epoch in range(epochs):
            epoch_g_loss = 0
            epoch_d_loss = 0
            epoch_l_loss = 0
            
            for batch_idx, (in_vitro_batch, in_vivo_batch) in enumerate(dataloader):
                # Train discriminator
                self.d_optimizer.zero_grad()
                
                # Real data
                real_labels = torch.ones(in_vitro_batch.size(0), 1).to(self.device)
                real_output = self.discriminator(in_vivo_batch, in_vitro_batch)
                d_real_loss = self.criterion(real_output, real_labels)
                
                # Fake data
                noise = torch.randn(in_vitro_batch.size(0), self.latent_dim).to(self.device)
                fake_data = self.generator(in_vitro_batch, noise)
                fake_labels = torch.zeros(in_vitro_batch.size(0), 1).to(self.device)
                fake_output = self.discriminator(fake_data.detach(), in_vitro_batch)
                d_fake_loss = self.criterion(fake_output, fake_labels)
                
                # Total discriminator loss
                d_loss = d_real_loss + d_fake_loss
                d_loss.backward()
                self.d_optimizer.step()
                
                # Train generator
                self.g_optimizer.zero_grad()
                
                noise = torch.randn(in_vitro_batch.size(0), self.latent_dim).to(self.device)
                fake_data = self.generator(in_vitro_batch, noise)
                fake_output = self.discriminator(fake_data, in_vitro_batch)
                
                # Generator tries to fool discriminator
                g_loss_adv = self.criterion(fake_output, real_labels)
                
                # Reconstruction loss
                g_loss_recon = self.mse_loss(fake_data, in_vivo_batch)
                
                # Total generator loss
                g_loss = g_loss_adv + 0.1 * g_loss_recon
                g_loss.backward()
                self.g_optimizer.step()
                
                # Train local optimizer
                self.l_optimizer.zero_grad()
                
                # Focus on key genes
                fake_data_key = fake_data[:, self.key_gene_indices]
                in_vivo_key = in_vivo_batch[:, self.key_gene_indices]
                l_pred = self.local_optimizer(fake_data)
                l_loss = self.mse_loss(l_pred, in_vivo_key)
                l_loss.backward()
                self.l_optimizer.step()
                
                epoch_g_loss += g_loss.item()
                epoch_d_loss += d_loss.item()
                epoch_l_loss += l_loss.item()
            
            # Calculate epoch averages
            num_batches = len(dataloader)
            epoch_g_loss /= num_batches
            epoch_d_loss /= num_batches
            epoch_l_loss /= num_batches
            
            # Calculate metrics
            if val_data is not None:
                cosine_sim = self.evaluate_cosine_similarity(val_data)
                pathway_acc = self.evaluate_pathway_accuracy(val_data)
                
                self.history['cosine_similarity'].append(cosine_sim)
                self.history['pathway_accuracy'].append(pathway_acc)
                
                logger.info(f"Epoch {epoch+1}/{epochs}: "
                          f"G Loss: {epoch_g_loss:.4f}, "
                          f"D Loss: {epoch_d_loss:.4f}, "
                          f"L Loss: {epoch_l_loss:.4f}, "
                          f"Cosine Sim: {cosine_sim:.4f}, "
                          f"Pathway Acc: {pathway_acc:.4f}")
            else:
                logger.info(f"Epoch {epoch+1}/{epochs}: "
                          f"G Loss: {epoch_g_loss:.4f}, "
                          f"D Loss: {epoch_d_loss:.4f}, "
                          f"L Loss: {epoch_l_loss:.4f}")
            
            self.history['g_loss'].append(epoch_g_loss)
            self.history['d_loss'].append(epoch_d_loss)
            self.history['l_loss'].append(epoch_l_loss)
        
        logger.info("AIVIVE model training completed")
    
    def predict(self, in_vitro_data: np.ndarray) -> np.ndarray:
        """
        Predict in vivo responses from in vitro data.
        
        Parameters
        ----------
        in_vitro_data : np.ndarray
            In vitro transcriptomic data
            
        Returns
        -------
        np.ndarray
            Predicted in vivo transcriptomic data
        """
        self.generator.eval()
        
        with torch.no_grad():
            in_vitro_tensor = torch.FloatTensor(in_vitro_data).to(self.device)
            noise = torch.randn(in_vitro_tensor.size(0), self.latent_dim).to(self.device)
            predictions = self.generator(in_vitro_tensor, noise)
            
        return predictions.cpu().numpy()
    
    def evaluate_cosine_similarity(self, data: Dict) -> float:
        """
        Evaluate cosine similarity between predictions and ground truth.
        
        Parameters
        ----------
        data : Dict
            Data with keys: 'in_vitro', 'in_vivo'
            
        Returns
        -------
        float
            Average cosine similarity
        """
        predictions = self.predict(data['in_vitro'])
        ground_truth = data['in_vivo']
        
        # Calculate cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity
        
        similarities = []
        for pred, true in zip(predictions, ground_truth):
            sim = cosine_similarity(pred.reshape(1, -1), true.reshape(1, -1))[0][0]
            similarities.append(sim)
        
        return np.mean(similarities)
    
    def evaluate_pathway_accuracy(self, data: Dict, 
                                  pathway_genes: List[List[int]] = None) -> float:
        """
        Evaluate accuracy for key toxicity pathways.
        
        Parameters
        ----------
        data : Dict
            Data with keys: 'in_vitro', 'in_vivo'
        pathway_genes : List[List[int]], optional
            List of gene indices for each pathway
            
        Returns
        -------
        float
            Pathway prediction accuracy
        """
        if pathway_genes is None:
            # Default to p53, apoptosis, ferroptosis pathways
            pathway_genes = [
                self.key_gene_indices[:100],  # p53 pathway
                self.key_gene_indices[100:200],  # apoptosis
                self.key_gene_indices[200:300]   # ferroptosis
            ]
        
        predictions = self.predict(data['in_vitro'])
        ground_truth = data['in_vivo']
        
        accuracies = []
        for pathway in pathway_genes:
            pred_pathway = predictions[:, pathway]
            true_pathway = ground_truth[:, pathway]
            
            # Calculate correlation
            corr_matrix = np.corrcoef(pred_pathway.T, true_pathway.T)
            pathway_acc = np.mean(np.diag(corr_matrix[:len(pathway), len(pathway):]))
            accuracies.append(pathway_acc)
        
        return np.mean(accuracies)
    
    def save(self, filepath: str):
        """
        Save model to file.
        
        Parameters
        ----------
        filepath : str
            Path to save model
        """
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'local_optimizer_state_dict': self.local_optimizer.state_dict(),
            'config': self.config,
            'history': self.history
        }, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """
        Load model from file.
        
        Parameters
        ----------
        filepath : str
            Path to load model from
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.local_optimizer.load_state_dict(checkpoint['local_optimizer_state_dict'])
        self.config = checkpoint['config']
        self.history = checkpoint['history']
        
        logger.info(f"Model loaded from {filepath}")