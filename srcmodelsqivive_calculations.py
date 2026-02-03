"""
QIVIVE (Quantitative In Vitro to In Vivo Extrapolation) calculations.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class QIVIVECalculator:
    """Calculate biophase concentrations and human equivalent doses."""
    
    def __init__(self, config: Dict):
        """
        Initialize QIVIVECalculator.
        
        Parameters
        ----------
        config : Dict
            QIVIVE configuration parameters
        """
        self.config = config
        
        # Default parameters
        self.ec50 = config.get('ec50', 50.0)  # μM
        self.fu = config.get('fu', 0.176)     # Free fraction
        self.k_cell = config.get('k_cell', 1.2)  # Cell partition coefficient
        self.hill_coefficient = config.get('hill_coefficient', 1.8)
        
        # PBPK model for reverse dosimetry
        self.pbpk_model = None
        
    def calculate_biophase_concentration(self, ec50: Optional[float] = None,
                                        fu: Optional[float] = None,
                                        k_cell: Optional[float] = None) -> float:
        """
        Calculate biophase concentration (Cbio) from in vitro EC50.
        
        Parameters
        ----------
        ec50 : float, optional
            In vitro EC50 (μM)
        fu : float, optional
            Free fraction in plasma
        k_cell : float, optional
            Cell partition coefficient
            
        Returns
        -------
        float
            Biophase concentration (μM)
        """
        # Use provided values or defaults
        ec50 = ec50 if ec50 is not None else self.ec50
        fu = fu if fu is not None else self.fu
        k_cell = k_cell if k_cell is not None else self.k_cell
        
        # Core QIVIVE equation: Cbio = EC50 × (fu / K)
        c_bio = ec50 * (fu / k_cell)
        
        logger.info(f"Calculated biophase concentration: "
                   f"Cbio = {ec50:.1f} × ({fu:.3f} / {k_cell:.3f}) = {c_bio:.2f} μM")
        
        return c_bio
    
    def calculate_concentration_dependent_cbio(self, ec50: float,
                                              concentration: float,
                                              binding_model: callable,
                                              partition_model: callable) -> float:
        """
        Calculate Cbio with concentration-dependent parameters.
        
        Parameters
        ----------
        ec50 : float
            In vitro EC50 (μM)
        concentration : float
            Estimated plasma concentration (μM)
        binding_model : callable
            Function returning fu for given concentration
        partition_model : callable
            Function returning K for given concentration
            
        Returns
        -------
        float
            Biophase concentration (μM)
        """
        # Get concentration-dependent parameters
        fu_dynamic = binding_model(concentration)
        k_dynamic = partition_model(concentration)
        
        # Calculate dynamic Cbio
        c_bio_dynamic = ec50 * (fu_dynamic / k_dynamic)
        
        logger.debug(f"Dynamic Cbio calculation: "
                    f"EC50={ec50:.1f} μM, C={concentration:.1f} μM, "
                    f"fu(C)={fu_dynamic:.3f}, K(C)={k_dynamic:.3f}, "
                    f"Cbio={c_bio_dynamic:.2f} μM")
        
        return c_bio_dynamic
    
    def calculate_human_equivalent_dose(self, c_bio: float,
                                       target_tissue: str = 'liver',
                                       route: str = 'oral',
                                       pbpk_model = None) -> float:
        """
        Calculate Human Equivalent Dose (HED) from biophase concentration.
        
        Parameters
        ----------
        c_bio : float
            Biophase concentration (μM)
        target_tissue : str, optional
            Target tissue for effect (default: 'liver')
        route : str, optional
            Administration route (default: 'oral')
        pbpk_model : object, optional
            PBPK model for reverse dosimetry
            
        Returns
        -------
        float
            Human Equivalent Dose (mg/kg)
        """
        logger.info(f"Calculating HED for Cbio={c_bio:.2f} μM in {target_tissue}")
        
        # Method 1: Use PBPK reverse dosimetry if available
        if pbpk_model is not None:
            self.pbpk_model = pbpk_model
            dose_mgkg = pbpk_model.reverse_dosimetry(
                target_concentration=c_bio,
                target_tissue=target_tissue,
                route=route
            )
            
            logger.info(f"PBPK reverse dosimetry: HED = {dose_mgkg:.4f} mg/kg")
            return dose_mgkg
        
        # Method 2: Simplified calculation using clearance
        # HED = (Cbio × CL) / F_abs
        clearance = self.config.get('clearance', 0.15)  # L/h/kg
        f_abs = self.config.get('absorption_fraction', 0.8)
        
        # Convert Cbio from μM to mg/L
        # Molecular weight of Nd(NO3)3·6H2O = 438.39 g/mol
        mw = 438.39  # g/mol
        c_bio_mgL = c_bio * mw / 1000  # Convert μM to mg/L
        
        # Calculate HED (mg/kg/day)
        # Assuming steady state: Dose = Css × CL / F
        hed = (c_bio_mgL * clearance * 24) / f_abs  # mg/kg/day
        
        logger.info(f"Simplified HED calculation: "
                   f"HED = ({c_bio_mgL:.2f} mg/L × {clearance:.2f} L/h/kg × 24 h) / {f_abs:.2f} "
                   f"= {hed:.4f} mg/kg/day")
        
        return hed
    
    def calculate_safety_margin(self, hed: float, 
                              reference_dose: Optional[float] = None) -> Dict:
        """
        Calculate safety margins and hazard quotients.
        
        Parameters
        ----------
        hed : float
            Human Equivalent Dose (mg/kg/day)
        reference_dose : float, optional
            Reference dose (RfD) for comparison
            
        Returns
        -------
        Dict
            Safety assessment metrics
        """
        if reference_dose is None:
            reference_dose = self.config.get('reference_dose', 0.25)  # mg/kg/day
        
        # Calculate Margin of Safety (MOS)
        mos = reference_dose / hed if hed > 0 else np.inf
        
        # Calculate Hazard Quotient (HQ)
        hq = hed / reference_dose
        
        # Risk classification
        if mos >= 100:
            risk_level = 'Very low risk'
            risk_color = 'green'
        elif mos >= 10:
            risk_level = 'Low risk'
            risk_color = 'blue'
        elif mos >= 1:
            risk_level = 'Medium risk'
            risk_color = 'yellow'
        else:
            risk_level = 'High risk'
            risk_color = 'red'
        
        safety_metrics = {
            'hed': hed,
            'reference_dose': reference_dose,
            'margin_of_safety': mos,
            'hazard_quotient': hq,
            'risk_level': risk_level,
            'risk_color': risk_color,
            'interpretation': f"MOS = {mos:.1f}x, {risk_level}"
        }
        
        logger.info(f"Safety assessment: MOS={mos:.1f}x, HQ={hq:.3f}, Risk={risk_level}")
        
        return safety_metrics
    
    def probabilistic_hed_calculation(self, parameter_distributions: Dict,
                                     n_iterations: int = 10000) -> Dict:
        """
        Perform probabilistic HED calculation using Monte Carlo simulation.
        
        Parameters
        ----------
        parameter_distributions : Dict
            Dictionary with parameter names and distribution functions
        n_iterations : int, optional
            Number of Monte Carlo iterations (default: 10000)
            
        Returns
        -------
        Dict
            Probabilistic HED results
        """
        logger.info(f"Starting probabilistic HED calculation with {n_iterations} iterations")
        
        # Latin Hypercube Sampling for better coverage
        from SALib.sample import latin
        
        # Define problem for SALib
        problem = {
            'num_vars': len(parameter_distributions),
            'names': list(parameter_distributions.keys()),
            'bounds': [[0, 1]] * len(parameter_distributions)  # Placeholder
        }
        
        # Generate samples
        param_samples = latin.sample(problem, n_iterations)
        
        # Convert to actual parameter values
        hed_samples = []
        
        for i in range(n_iterations):
            # Sample parameters from their distributions
            params = {}
            for j, (param_name, dist_func) in enumerate(parameter_distributions.items()):
                # dist_func should be a function that returns a sampled value
                params[param_name] = dist_func()
            
            # Calculate HED for this parameter set
            c_bio = self.calculate_biophase_concentration(
                ec50=params.get('ec50', self.ec50),
                fu=params.get('fu', self.fu),
                k_cell=params.get('k_cell', self.k_cell)
            )
            
            hed = self.calculate_human_equivalent_dose(c_bio)
            hed_samples.append(hed)
        
        hed_samples = np.array(hed_samples)
        
        # Calculate statistics
        hed_stats = {
            'samples': hed_samples,
            'mean': np.mean(hed_samples),
            'median': np.median(hed_samples),
            'std': np.std(hed_samples),
            'cv': np.std(hed_samples) / np.mean(hed_samples) if np.mean(hed_samples) > 0 else 0,
            'percentiles': {
                '2.5': np.percentile(hed_samples, 2.5),
                '25': np.percentile(hed_samples, 25),
                '50': np.percentile(hed_samples, 50),
                '75': np.percentile(hed_samples, 75),
                '97.5': np.percentile(hed_samples, 97.5)
            },
            'risk_probabilities': {
                'very_low': np.mean(hed_samples < 0.01),
                'low': np.mean((hed_samples >= 0.01) & (hed_samples < 0.024)),
                'medium': np.mean((hed_samples >= 0.024) & (hed_samples < 0.037)),
                'high': np.mean((hed_samples >= 0.037) & (hed_samples < 0.1)),
                'very_high': np.mean(hed_samples >= 0.1)
            }
        }
        
        logger.info(f"Probabilistic HED results: "
                   f"Median={hed_stats['median']:.4f} mg/kg/day, "
                   f"95% CI=[{hed_stats['percentiles']['2.5']:.4f}, "
                   f"{hed_stats['percentiles']['97.5']:.4f}]")
        
        return hed_stats
    
    def exposure_scenario_analysis(self, exposure_scenarios: Dict) -> pd.DataFrame:
        """
        Analyze different exposure scenarios.
        
        Parameters
        ----------
        exposure_scenarios : Dict
            Dictionary of exposure scenarios with parameters
            
        Returns
        -------
        pd.DataFrame
            Risk assessment for each scenario
        """
        logger.info(f"Analyzing {len(exposure_scenarios)} exposure scenarios")
        
        results = []
        
        for scenario_name, scenario_params in exposure_scenarios.items():
            # Calculate daily intake
            if 'air_concentration' in scenario_params:
                # Inhalation exposure
                air_conc = scenario_params['air_concentration']  # mg/m³
                inhalation_rate = scenario_params.get('inhalation_rate', 20)  # m³/day
                absorption = scenario_params.get('absorption_fraction', 0.8)
                
                daily_intake = air_conc * inhalation_rate * absorption / self.config.get('body_weight', 70.0)
                exposure_route = 'inhalation'
                
            elif 'water_concentration' in scenario_params:
                # Oral exposure via water
                water_conc = scenario_params['water_concentration']  # mg/L
                water_intake = scenario_params.get('water_intake', 2.0)  # L/day
                absorption = scenario_params.get('absorption_fraction', 0.8)
                
                daily_intake = water_conc * water_intake * absorption / self.config.get('body_weight', 70.0)
                exposure_route = 'oral'
            else:
                # Direct dose
                daily_intake = scenario_params.get('dose', 0)
                exposure_route = scenario_params.get('route', 'oral')
            
            # Calculate HED (simplified - in practice would use PBPK)
            # This assumes linear kinetics
            hed = daily_intake
            
            # Calculate safety metrics
            safety = self.calculate_safety_margin(hed)
            
            # Risk multiplier (relative to reference dose)
            risk_multiplier = hed / safety['reference_dose']
            
            # Determine risk level based on multiplier
            if risk_multiplier < 0.05:
                risk_category = 'Very low risk'
                action = 'No action required'
            elif risk_multiplier < 0.1:
                risk_category = 'Low risk'
                action = 'Routine monitoring'
            elif risk_multiplier < 1.0:
                risk_category = 'Medium risk'
                action = 'Enhanced controls'
            elif risk_multiplier < 10.0:
                risk_category = 'High risk'
                action = 'Immediate controls'
            else:
                risk_category = 'Very high risk'
                action = 'Emergency response'
            
            results.append({
                'Scenario': scenario_name,
                'Exposure Route': exposure_route,
                'Daily Intake (mg/kg/day)': f"{daily_intake:.6f}",
                'HED (mg/kg/day)': f"{hed:.6f}",
                'Risk Multiplier': f"{risk_multiplier:.2f}×",
                'Margin of Safety': f"{safety['margin_of_safety']:.1f}×",
                'Risk Category': risk_category,
                'Recommended Action': action
            })
        
        return pd.DataFrame(results)