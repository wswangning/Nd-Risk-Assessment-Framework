"""
PBPK (Physiologically Based Pharmacokinetic) model with nonlinear kinetics.
"""

import numpy as np
from scipy.integrate import odeint
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class NonlinearPBPKModel:
    """PBPK model with nonlinear protein binding and concentration-dependent partitioning."""
    
    def __init__(self, config: Dict):
        """
        Initialize NonlinearPBPKModel.
        
        Parameters
        ----------
        config : Dict
            Model configuration parameters
        """
        self.config = config
        
        # Physiological parameters
        self.body_weight = config.get('body_weight', 70.0)  # kg
        self.compartments = config.get('compartments', ['plasma', 'liver', 'kidney', 'brain', 'rest'])
        
        # Volume parameters (L)
        self.volumes = {
            'plasma': config.get('volume_plasma', 3.0),
            'liver': config.get('volume_liver', 1.8),
            'kidney': config.get('volume_kidney', 0.3),
            'brain': config.get('volume_brain', 1.4),
            'rest': config.get('volume_rest', 63.5)
        }
        
        # Blood flow parameters (L/h)
        self.blood_flows = {
            'liver': config.get('blood_flow_liver', 25.0),
            'kidney': config.get('blood_flow_kidney', 19.0),
            'brain': config.get('blood_flow_brain', 15.0),
            'rest': config.get('blood_flow_rest', 46.0)
        }
        
        # Clearance parameters (L/h)
        self.clearances = {
            'hepatic': config.get('clearance_hepatic', 10.5),
            'renal': config.get('clearance_renal', 3.5)
        }
        
        # Nonlinear binding parameters
        self.nonlinear_binding = config.get('nonlinear_binding', True)
        self.b_max = config.get('b_max', 150.2)  # μM
        self.k_d = config.get('k_d', 12.5)       # μM
        
        # Concentration-dependent partition coefficients
        self.k_values = {
            'liver': config.get('k_liver', 8.2),
            'kidney': config.get('k_kidney', 5.6),
            'brain': config.get('k_brain', 0.12),
            'rest': config.get('k_rest', 1.0)
        }
        
        # Absorption parameters
        self.ka = config.get('absorption_rate', 0.5)  # 1/h
        self.f_abs = config.get('absorption_fraction', 0.8)
        
    def nonlinear_protein_binding(self, concentration: float) -> float:
        """
        Calculate free fraction (fu) based on nonlinear U-shaped binding.
        
        Parameters
        ----------
        concentration : float
            Total concentration (μM)
            
        Returns
        -------
        float
            Free fraction (fu)
        """
        if not self.nonlinear_binding:
            return 0.176  # Constant value for mid-concentration
        
        # U-shaped binding model
        # fu = 1 / (1 + Bmax/(Kd + C))
        if concentration <= 0:
            return 1.0
        
        fu = 1.0 / (1.0 + self.b_max / (self.k_d + concentration))
        
        # Ensure fu stays within reasonable bounds
        return np.clip(fu, 0.01, 0.99)
    
    def concentration_dependent_partition(self, concentration: float, 
                                         tissue: str = 'liver') -> float:
        """
        Calculate concentration-dependent partition coefficient.
        
        Parameters
        ----------
        concentration : float
            Concentration in tissue (μM)
        tissue : str, optional
            Tissue type (default: 'liver')
            
        Returns
        -------
        float
            Partition coefficient (K)
        """
        base_k = self.k_values.get(tissue, 1.0)
        
        # Inverted U-shaped concentration dependence
        # Based on experimental data for Nd
        if concentration <= 0:
            return base_k
        
        # Simplified model: K decreases at very high concentrations
        if concentration > 100:  # μM
            return base_k * 0.5
        elif concentration > 10:
            return base_k * 1.2
        else:
            return base_k
    
    def model_equations(self, y: np.ndarray, t: float, dose: float, 
                       route: str = 'oral') -> np.ndarray:
        """
        Define PBPK model differential equations.
        
        Parameters
        ----------
        y : np.ndarray
            Current state variables
        t : float
            Current time
        dose : float
            Administered dose (mg/kg)
        route : str, optional
            Administration route (default: 'oral')
            
        Returns
        -------
        np.ndarray
            Derivatives of state variables
        """
        # Unpack state variables
        # y = [A_plasma, A_liver, A_kidney, A_brain, A_rest, A_gi] for oral
        if route == 'oral':
            A_plasma, A_liver, A_kidney, A_brain, A_rest, A_gi = y
        else:  # iv
            A_plasma, A_liver, A_kidney, A_brain, A_rest = y
            A_gi = 0
        
        # Calculate concentrations (μM)
        # Assuming molecular weight of Nd(NO3)3·6H2O = 438.39 g/mol
        mw = 438.39  # g/mol
        
        C_plasma = (A_plasma / self.volumes['plasma']) * 1000 / mw  # Convert to μM
        C_liver = (A_liver / self.volumes['liver']) * 1000 / mw
        C_kidney = (A_kidney / self.volumes['kidney']) * 1000 / mw
        C_brain = (A_brain / self.volumes['brain']) * 1000 / mw
        
        # Calculate free fractions
        fu_plasma = self.nonlinear_protein_binding(C_plasma)
        
        # Calculate partition coefficients (concentration-dependent)
        K_liver = self.concentration_dependent_partition(C_liver, 'liver')
        K_kidney = self.concentration_dependent_partition(C_kidney, 'kidney')
        K_brain = self.concentration_dependent_partition(C_brain, 'brain')
        K_rest = self.concentration_dependent_partition(0, 'rest')  # Assume low concentration
        
        # Calculate venous concentrations (assuming well-stirred model)
        Cv_liver = C_liver / K_liver
        Cv_kidney = C_kidney / K_kidney
        Cv_brain = C_brain / K_brain
        Cv_rest = (A_rest / (self.volumes['rest'] * K_rest)) * 1000 / mw
        
        # Arterial concentration (mixed venous return)
        Q_total = sum(self.blood_flows.values())
        Ca = (self.blood_flows['liver'] * Cv_liver +
              self.blood_flows['kidney'] * Cv_kidney +
              self.blood_flows['brain'] * Cv_brain +
              self.blood_flows['rest'] * Cv_rest) / Q_total
        
        # Differential equations
        dA_dt = np.zeros_like(y)
        
        if route == 'oral':
            # Gastrointestinal tract
            dA_dt[5] = -self.ka * A_gi  # A_gi
            
            # Plasma compartment
            dA_dt[0] = (self.ka * A_gi * self.f_abs +  # Absorption from GI
                       self.blood_flows['liver'] * (Cv_liver - Ca) +
                       self.blood_flows['kidney'] * (Cv_kidney - Ca) +
                       self.blood_flows['brain'] * (Cv_brain - Ca) +
                       self.blood_flows['rest'] * (Cv_rest - Ca))
            
            # Liver compartment (includes hepatic clearance)
            dA_dt[1] = (self.blood_flows['liver'] * (Ca - Cv_liver) -
                       self.clearances['hepatic'] * C_plasma * fu_plasma)
            
            # Kidney compartment (includes renal clearance)
            dA_dt[2] = (self.blood_flows['kidney'] * (Ca - Cv_kidney) -
                       self.clearances['renal'] * C_plasma * fu_plasma)
            
            # Brain compartment
            dA_dt[3] = self.blood_flows['brain'] * (Ca - Cv_brain)
            
            # Rest of body compartment
            dA_dt[4] = self.blood_flows['rest'] * (Ca - Cv_rest)
            
        else:  # iv administration
            # IV bolus input
            iv_input = dose * self.body_weight * 1000 / mw if t == 0 else 0  # μM
            
            # Plasma compartment
            dA_dt[0] = (iv_input +
                       self.blood_flows['liver'] * (Cv_liver - Ca) +
                       self.blood_flows['kidney'] * (Cv_kidney - Ca) +
                       self.blood_flows['brain'] * (Cv_brain - Ca) +
                       self.blood_flows['rest'] * (Cv_rest - Ca))
            
            # Other compartments (same as oral)
            dA_dt[1] = (self.blood_flows['liver'] * (Ca - Cv_liver) -
                       self.clearances['hepatic'] * C_plasma * fu_plasma)
            dA_dt[2] = (self.blood_flows['kidney'] * (Ca - Cv_kidney) -
                       self.clearances['renal'] * C_plasma * fu_plasma)
            dA_dt[3] = self.blood_flows['brain'] * (Ca - Cv_brain)
            dA_dt[4] = self.blood_flows['rest'] * (Ca - Cv_rest)
        
        return dA_dt
    
    def simulate(self, dose: float, route: str = 'oral', 
                duration: float = 72.0, time_points: int = 100) -> Dict:
        """
        Simulate PBPK model for given dose.
        
        Parameters
        ----------
        dose : float
            Administered dose (mg/kg)
        route : str, optional
            Administration route (default: 'oral')
        duration : float, optional
            Simulation duration in hours (default: 72)
        time_points : int, optional
            Number of time points (default: 100)
            
        Returns
        -------
        Dict
            Simulation results including time course and AUC
        """
        logger.info(f"Simulating PBPK model: dose={dose} mg/kg, route={route}")
        
        # Time points
        t = np.linspace(0, duration, time_points)
        
        # Initial conditions
        if route == 'oral':
            # Initial dose in GI tract
            initial_dose_mg = dose * self.body_weight
            initial_dose_mmol = initial_dose_mg / 438.39  # mmol
            
            y0 = [0, 0, 0, 0, 0, initial_dose_mmol]  # Add GI compartment
        else:  # iv
            y0 = [0, 0, 0, 0, 0]
        
        # Solve ODEs
        solution = odeint(
            lambda y, t: self.model_equations(y, t, dose, route),
            y0, t
        )
        
        # Calculate concentrations (μM)
        mw = 438.39  # g/mol
        
        if route == 'oral':
            C_plasma = (solution[:, 0] / self.volumes['plasma']) * 1000 / mw
            C_liver = (solution[:, 1] / self.volumes['liver']) * 1000 / mw
            C_kidney = (solution[:, 2] / self.volumes['kidney']) * 1000 / mw
            C_brain = (solution[:, 3] / self.volumes['brain']) * 1000 / mw
            C_rest = (solution[:, 4] / self.volumes['rest']) * 1000 / mw
        else:
            C_plasma = (solution[:, 0] / self.volumes['plasma']) * 1000 / mw
            C_liver = (solution[:, 1] / self.volumes['liver']) * 1000 / mw
            C_kidney = (solution[:, 2] / self.volumes['kidney']) * 1000 / mw
            C_brain = (solution[:, 3] / self.volumes['brain']) * 1000 / mw
            C_rest = (solution[:, 4] / self.volumes['rest']) * 1000 / mw
        
        # Calculate AUC using trapezoidal rule
        auc_plasma = np.trapz(C_plasma, t)
        auc_liver = np.trapz(C_liver, t)
        auc_kidney = np.trapz(C_kidney, t)
        
        # Calculate Cmax
        cmax_plasma = np.max(C_plasma)
        cmax_liver = np.max(C_liver)
        
        # Calculate free concentrations
        fu_values = np.array([self.nonlinear_protein_binding(c) for c in C_plasma])
        C_free = C_plasma * fu_values
        
        results = {
            'time': t,
            'concentrations': {
                'plasma': C_plasma,
                'liver': C_liver,
                'kidney': C_kidney,
                'brain': C_brain,
                'rest': C_rest,
                'free': C_free
            },
            'auc': {
                'plasma': auc_plasma,
                'liver': auc_liver,
                'kidney': auc_kidney
            },
            'cmax': {
                'plasma': cmax_plasma,
                'liver': cmax_liver
            },
            'fu_profiles': fu_values,
            'dose': dose,
            'route': route
        }
        
        return results
    
    def reverse_dosimetry(self, target_concentration: float, 
                         target_tissue: str = 'liver',
                         route: str = 'oral') -> float:
        """
        Perform reverse dosimetry to find dose that yields target concentration.
        
        Parameters
        ----------
        target_concentration : float
            Target concentration in tissue (μM)
        target_tissue : str, optional
            Target tissue (default: 'liver')
        route : str, optional
            Administration route (default: 'oral')
            
        Returns
        -------
        float
            Estimated dose (mg/kg) to achieve target concentration
        """
        logger.info(f"Performing reverse dosimetry for C_target={target_concentration} μM in {target_tissue}")
        
        # Use binary search to find dose
        dose_low = 0.001  # mg/kg
        dose_high = 1000.0  # mg/kg
        tolerance = 0.01  # 1% tolerance
        
        for _ in range(50):  # Maximum 50 iterations
            dose_mid = (dose_low + dose_high) / 2
            
            # Simulate with mid dose
            results = self.simulate(dose_mid, route, duration=24, time_points=50)
            
            # Get peak concentration in target tissue
            C_peak = np.max(results['concentrations'][target_tissue])
            
            # Check if we found the dose
            if abs(C_peak - target_concentration) / target_concentration < tolerance:
                return dose_mid
            
            # Adjust search bounds
            if C_peak < target_concentration:
                dose_low = dose_mid
            else:
                dose_high = dose_mid
        
        # Return best estimate
        return (dose_low + dose_high) / 2