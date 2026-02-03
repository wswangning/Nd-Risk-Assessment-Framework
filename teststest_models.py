"""
Unit tests for model implementations.
"""

import unittest
import numpy as np
import torch
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.aivive_model import Generator, Discriminator, AIVIVEModel
from src.models.pbpk_model import NonlinearPBPKModel
from src.models.qivive_calculations import QIVIVECalculator

class TestAIVIVEModels(unittest.TestCase):
    """Test AIVIVE model components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.input_dim = 100
        self.latent_dim = 20
        self.batch_size = 16
        
    def test_generator_creation(self):
        """Test Generator model creation."""
        generator = Generator(self.input_dim, self.latent_dim)
        
        # Check model structure
        self.assertIsInstance(generator, torch.nn.Module)
        
        # Test forward pass
        x = torch.randn(self.batch_size, self.input_dim)
        z = torch.randn(self.batch_size, self.latent_dim)
        output = generator(x, z)
        
        self.assertEqual(output.shape, (self.batch_size, self.input_dim))
        
    def test_discriminator_creation(self):
        """Test Discriminator model creation."""
        discriminator = Discriminator(self.input_dim)
        
        # Check model structure
        self.assertIsInstance(discriminator, torch.nn.Module)
        
        # Test forward pass
        x = torch.randn(self.batch_size, self.input_dim)
        condition = torch.randn(self.batch_size, self.input_dim)
        output = discriminator(x, condition)
        
        self.assertEqual(output.shape, (self.batch_size, 1))
        # Output should be between 0 and 1 (probability)
        self.assertTrue(torch.all(output >= 0))
        self.assertTrue(torch.all(output <= 1))
    
    def test_aivive_model_initialization(self):
        """Test AIVIVE model initialization."""
        config = {
            'input_dim': self.input_dim,
            'latent_dim': self.latent_dim,
            'key_gene_indices': list(range(20)),
            'learning_rate': 0.0002
        }
        
        model = AIVIVEModel(config, device='cpu')
        
        # Check components
        self.assertIsInstance(model.generator, Generator)
        self.assertIsInstance(model.discriminator, Discriminator)
        self.assertIsInstance(model.local_optimizer, torch.nn.Module)
        
        # Check optimizers
        self.assertIsInstance(model.g_optimizer, torch.optim.Adam)
        self.assertIsInstance(model.d_optimizer, torch.optim.Adam)
        self.assertIsInstance(model.l_optimizer, torch.optim.Adam)

class TestPBPKModel(unittest.TestCase):
    """Test PBPK model."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'body_weight': 70.0,
            'volume_plasma': 3.0,
            'volume_liver': 1.8,
            'volume_kidney': 0.3,
            'volume_brain': 1.4,
            'volume_rest': 63.5,
            'nonlinear_binding': True,
            'b_max': 150.2,
            'k_d': 12.5
        }
        
    def test_model_initialization(self):
        """Test PBPK model initialization."""
        model = NonlinearPBPKModel(self.config)
        
        # Check parameters
        self.assertEqual(model.body_weight, 70.0)
        self.assertEqual(model.b_max, 150.2)
        self.assertEqual(model.k_d, 12.5)
        self.assertTrue(model.nonlinear_binding)
        
    def test_nonlinear_protein_binding(self):
        """Test nonlinear protein binding calculation."""
        model = NonlinearPBPKModel(self.config)
        
        # Test at different concentrations
        concentrations = [1.0, 10.0, 100.0, 1000.0]
        
        for conc in concentrations:
            fu = model.nonlinear_protein_binding(conc)
            
            # fu should be between 0 and 1
            self.assertGreaterEqual(fu, 0.0)
            self.assertLessEqual(fu, 1.0)
            
            # At very high concentrations, fu should approach 1
            if conc > 1000:
                self.assertGreater(fu, 0.9)
    
    def test_simulation(self):
        """Test PBPK simulation."""
        model = NonlinearPBPKModel(self.config)
        
        # Run simulation
        results = model.simulate(dose=10.0, route='oral', duration=24.0, time_points=50)
        
        # Check results structure
        expected_keys = ['time', 'concentrations', 'auc', 'cmax', 'fu_profiles', 'dose', 'route']
        for key in expected_keys:
            self.assertIn(key, results)
        
        # Check array shapes
        self.assertEqual(len(results['time']), 50)
        self.assertEqual(len(results['concentrations']['plasma']), 50)
        self.assertEqual(len(results['concentrations']['liver']), 50)
        
        # AUC should be positive
        self.assertGreater(results['auc']['plasma'], 0)
        self.assertGreater(results['auc']['liver'], 0)

class TestQIVIVECalculator(unittest.TestCase):
    """Test QIVIVE calculator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'ec50': 50.0,
            'fu': 0.176,
            'k_cell': 1.2,
            'hill_coefficient': 1.8,
            'clearance': 0.15,
            'absorption_fraction': 0.8,
            'reference_dose': 0.25
        }
        
    def test_calculate_biophase_concentration(self):
        """Test biophase concentration calculation."""
        calculator = QIVIVECalculator(self.config)
        
        # Test with default parameters
        c_bio = calculator.calculate_biophase_concentration()
        expected = 50.0 * (0.176 / 1.2)
        self.assertAlmostEqual(c_bio, expected, places=5)
        
        # Test with custom parameters
        c_bio_custom = calculator.calculate_biophase_concentration(
            ec50=100.0, fu=0.5, k_cell=2.0
        )
        expected_custom = 100.0 * (0.5 / 2.0)
        self.assertAlmostEqual(c_bio_custom, expected_custom, places=5)
    
    def test_calculate_safety_margin(self):
        """Test safety margin calculation."""
        calculator = QIVIVECalculator(self.config)
        
        # Test with different HED values
        test_cases = [
            (0.01, 25.0),   # Very safe
            (0.1, 2.5),     # Marginally safe
            (0.25, 1.0),    # At reference dose
            (0.5, 0.5),     # Unsafe
        ]
        
        for hed, expected_mos in test_cases:
            safety = calculator.calculate_safety_margin(hed)
            self.assertAlmostEqual(safety['margin_of_safety'], expected_mos, places=2)
            
            # Check risk level classification
            if hed < 0.01:
                self.assertEqual(safety['risk_level'], 'Very low risk')
            elif hed < 0.024:
                self.assertEqual(safety['risk_level'], 'Low risk')
            elif hed < 0.037:
                self.assertEqual(safety['risk_level'], 'Medium risk')
            else:
                self.assertEqual(safety['risk_level'], 'High risk')
    
    def test_probabilistic_calculation(self):
        """Test probabilistic HED calculation."""
        calculator = QIVIVECalculator(self.config)
        
        # Define parameter distributions
        parameter_distributions = {
            'ec50': lambda: np.random.lognormal(mean=np.log(50.0), sigma=0.3),
            'fu': lambda: np.random.normal(loc=0.176, scale=0.05),
            'k_cell': lambda: np.random.normal(loc=1.2, scale=0.2)
        }
        
        # Run probabilistic calculation
        results = calculator.probabilistic_hed_calculation(
            parameter_distributions, n_iterations=100
        )
        
        # Check results structure
        expected_keys = ['samples', 'mean', 'median', 'std', 'cv', 'percentiles', 'risk_probabilities']
        for key in expected_keys:
            self.assertIn(key, results)
        
        # Check sample size
        self.assertEqual(len(results['samples']), 100)
        
        # Check percentiles
        percentiles = results['percentiles']
        self.assertLess(percentiles['2.5'], percentiles['50'])
        self.assertLess(percentiles['50'], percentiles['97.5'])

if __name__ == '__main__':
    unittest.main()