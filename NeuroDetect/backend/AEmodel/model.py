import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

class FraudAutoencoder(nn.Module):
    """Autoencoder model for fraud detection"""
    
    def __init__(self, input_dim, hidden_dim1=128, hidden_dim2=64, 
                 latent_dim=16, dropout_rate=0.000287):
        super(FraudAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim2, latent_dim),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim2, hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim1, input_dim)
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def get_reconstruction_error(self, x):
        """Calculate reconstruction error"""
        with torch.no_grad():
            reconstructed = self.forward(x)
            error = torch.mean((x - reconstructed) ** 2, dim=1)
        return error.cpu().numpy()

class EarlyStopping:
    """Early stopping utility"""
    
    def __init__(self, patience=10, min_delta=0, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_state = None
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0
    
    def save_checkpoint(self, model):
        if self.verbose:
            print(f'Validation loss decreased to {self.best_loss:.6f}. Saving model...')
        self.best_model_state = model.state_dict().copy()
    
    def load_best_model(self, model):
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)

class AutoencoderTrainer:
    """Trainer for the autoencoder model"""
    
    def __init__(self, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.train_losses = []
        self.val_losses = []
        
    def train(self, X_train, X_val, input_dim, epochs=100, 
              batch_size=128, patience=10):
        """
        Train the autoencoder
        
        Args:
            X_train: Training data
            X_val: Validation data
            input_dim: Input dimension
            epochs: Maximum epochs
            batch_size: Batch size
            patience: Early stopping patience
            
        Returns:
            Trained model
        """
        # Create data loaders
        train_dataset = TensorDataset(torch.FloatTensor(X_train))
        val_dataset = TensorDataset(torch.FloatTensor(X_val))
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize model
        self.model = FraudAutoencoder(
            input_dim=input_dim,
            hidden_dim1=128,
            hidden_dim2=64,
            latent_dim=16,
            dropout_rate=0.000287
        ).to(self.device)
        
        # Loss and optimizer
        criterion = nn.HuberLoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=0.000608939,
            weight_decay=1.5073e-05
        )
        
        # Early stopping
        early_stopping = EarlyStopping(patience=patience, verbose=True)
        
        print("Training Autoencoder...")
        print(f"Device: {self.device}")
        
        self.train_losses = []
        self.val_losses = []
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            for batch in train_loader:
                data = batch[0].to(self.device)
                
                optimizer.zero_grad()
                reconstructed = self.model(data)
                loss = criterion(reconstructed, data)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    data = batch[0].to(self.device)
                    reconstructed = self.model(data)
                    loss = criterion(reconstructed, data)
                    val_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            self.train_losses.append(avg_train_loss)
            self.val_losses.append(avg_val_loss)
            
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f'Epoch [{epoch+1}/{epochs}], '
                      f'Train Loss: {avg_train_loss:.6f}, '
                      f'Val Loss: {avg_val_loss:.6f}')
            
            # Early stopping check
            early_stopping(avg_val_loss, self.model)
            if early_stopping.early_stop:
                print(f'\nEarly stopping triggered at epoch {epoch+1}')
                break
        
        # Load best model
        early_stopping.load_best_model(self.model)
        print(f'\nTraining completed!')
        
        return self.model
    
    def save_model(self, path):
        """Save model to file"""
        if self.model:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'input_dim': self.model.encoder[0].in_features,
                'train_losses': self.train_losses,
                'val_losses': self.val_losses
            }, path)
    
    def load_model(self, path, device=None):
        """Load model from file"""
        checkpoint = torch.load(path, map_location=device or self.device)
        self.model = FraudAutoencoder(
            input_dim=checkpoint['input_dim'],
            hidden_dim1=128,
            hidden_dim2=64,
            latent_dim=16,
            dropout_rate=0.000287
        ).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        return self.model