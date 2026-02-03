#!/usr/bin/env python
"""
Main pipeline script for complete risk assessment workflow.
"""

import argparse
import yaml
import json
import logging
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_processing.preprocess_omics import OmicsDataProcessor
from src.models.aivive_model import AIVIVEModel
from src.models.pbpk_model import NonlinearPBPKModel
from src.models.httk_calibration import HTTKCalibrator
from src.models.qivive_calculations import QIVIVECalculator
from src.analysis.risk_assessment import ProbabilisticRiskAssessment
from src.visualization.plot_results import RiskAssessmentVisualizer

def setup_logging(output_dir: Path):
    """Setup logging configuration."""
    log_dir = output_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'pipeline.log'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def main():
    """Main pipeline function."""
    parser = argparse.ArgumentParser(
        description='Run complete AIVIVE-PBPK-QIVIVE risk assessment pipeline'
    )
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration YAML file')
    parser.add_argument('--output', type=str, default='results',
                       help='Output directory for results')
    parser.add_argument('--skip-aivive', action='store_true',
                       help='Skip AIVIVE model training (use pretrained)')
    parser.add_argument('--skip-httk', action='store_true',
                       help='Skip HTTK calibration')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(output_dir)
    logger.info("Starting complete risk assessment pipeline")
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Output directory: {output_dir}")
    
    # Load configuration
    config = load_config(args.config)
    
    try:
        # Step 1: Data processing
        logger.info("Step 1: Data processing and integration")
        data_processor = OmicsDataProcessor(config.get('data_processing', {}))
        
        # Load omics data (example paths - should be configured)
        transcriptomics_path = config.get('data_paths', {}).get('transcriptomics')
        metabolomics_path = config.get('data_paths', {}).get('metabolomics')
        
        if transcriptomics_path:
            transcriptomics = data_processor.load_transcriptomics(transcriptomics_path)
        
        if metabolomics_path:
            metabolomics = data_processor.load_metabolomics(metabolomics_path)
        
        # Step 2: HTTK calibration
        if not args.skip_httk:
            logger.info("Step 2: HTTK model calibration")
            
            httk_calibrator = HTTKCalibrator(config.get('httk', {}))
            
            # Load experimental data (example)
            experimental_data = {
                'plasma_protein_binding': {
                    'concentrations': [1.0, 10.0, 100.0],
                    'fu_values': [0.556, 0.176, 0.965]
                },
                'partition_coefficients': {
                    'concentrations': [5.0, 50.0, 250.0],
                    'k_values': [0.048, 0.079, 0.034]
                },
                'clearance': 0.15
            }
            
            httk_calibrator.load_experimental_data(experimental_data)
            
            # Calibrate models
            plasma_calibration = httk_calibrator.calibrate_plasma_protein_binding()
            partition_calibration = httk_calibrator.calibrate_partition_coefficients()
            
            # Save calibration results
            calibration_results = {
                'plasma_binding': plasma_calibration,
                'partition': partition_calibration
            }
            
            with open(output_dir / 'httk_calibration.json', 'w') as f:
                json.dump(calibration_results, f, indent=2, default=str)
        
        # Step 3: AIVIVE model training
        if not args.skip_aivive:
            logger.info("Step 3: AIVIVE model training")
            
            aivive_config = config.get('aivive', {})
            aivive_model = AIVIVEModel(aivive_config, device='cpu')
            
            # Example training data (should be loaded from actual data)
            # This is just a placeholder
            n_samples = 100
            n_genes = aivive_config.get('input_dim', 15000)
            
            train_data = {
                'in_vitro': np.random.randn(n_samples, n_genes),
                'in_vivo': np.random.randn(n_samples, n_genes)
            }
            
            val_data = {
                'in_vitro': np.random.randn(20, n_genes),
                'in_vivo': np.random.randn(20, n_genes)
            }
            
            # Train model
            aivive_model.train(
                train_data=train_data,
                val_data=val_data,
                epochs=aivive_config.get('epochs', 100),
                batch_size=aivive_config.get('batch_size', 32)
            )
            
            # Save trained model
            model_path = output_dir / 'aivive_model.pth'
            aivive_model.save(str(model_path))
        
        # Step 4: PBPK model setup
        logger.info("Step 4: PBPK model setup")
        
        pbpk_config = config.get('pbpk', {})
        pbpk_model = NonlinearPBPKModel(pbpk_config)
        
        # Step 5: QIVIVE calculations
        logger.info("Step 5: QIVIVE calculations")
        
        qivive_config = config.get('qivive', {})
        qivive_calculator = QIVIVECalculator(qivive_config)
        
        # Calculate biophase concentration
        c_bio = qivive_calculator.calculate_biophase_concentration()
        
        # Calculate HED using PBPK reverse dosimetry
        hed = qivive_calculator.calculate_human_equivalent_dose(
            c_bio=c_bio,
            target_tissue='liver',
            route='oral',
            pbpk_model=pbpk_model
        )
        
        # Calculate safety margins
        safety = qivive_calculator.calculate_safety_margin(hed)
        
        # Step 6: Probabilistic risk assessment
        logger.info("Step 6: Probabilistic risk assessment")
        
        risk_config = config.get('risk', {})
        risk_assessor = ProbabilisticRiskAssessment(risk_config)
        
        # Define parameter distributions for Monte Carlo
        parameter_distributions = {
            'ec50': {
                'distribution': 'lognormal',
                'mean': np.log(50.0),
                'std': 0.3,
                'min': 10.0,
                'max': 200.0
            },
            'fu': {
                'distribution': 'normal',
                'mean': 0.176,
                'std': 0.05,
                'min': 0.01,
                'max': 0.99
            },
            'k_cell': {
                'distribution': 'normal',
                'mean': 1.2,
                'std': 0.2,
                'min': 0.5,
                'max': 2.0
            },
            'clearance': {
                'distribution': 'uniform',
                'min': 0.08,
                'max': 0.25
            }
        }
        
        # Generate parameter samples
        param_samples = risk_assessor.latin_hypercube_sampling(
            parameter_distributions,
            n_samples=1000
        )
        
        # Define model function for Monte Carlo
        def hed_model(ec50, fu, k_cell, clearance):
            c_bio = ec50 * (fu / k_cell)
            hed_simple = (c_bio * 438.39 / 1000 * clearance * 24) / 0.8
            return hed_simple
        
        # Run Monte Carlo simulation
        mc_results = risk_assessor.monte_carlo_simulation(
            model_function=hed_model,
            parameter_samples=param_samples
        )
        
        # Calculate risk probabilities
        risk_results = risk_assessor.calculate_risk_probabilities(
            mc_results['hed'].values
        )
        
        # Sensitivity analysis
        sensitivity_results = risk_assessor.sensitivity_analysis_sobol(
            model_function=hed_model,
            parameter_distributions=parameter_distributions,
            n_samples=500
        )
        
        # Population exposure analysis
        exposure_scenarios = {
            'general_population': {
                'distribution': 'lognormal',
                'log_mean': np.log(0.65e-3),  # 0.65 μg/kg/day = 0.00065 mg/kg/day
                'log_std': 0.5
            },
            'mining_area_residents': {
                'distribution': 'lognormal',
                'log_mean': np.log(16.7e-3),  # 16.7 μg/kg/day = 0.0167 mg/kg/day
                'log_std': 0.5
            }
        }
        
        population_results = risk_assessor.population_exposure_analysis(
            hed_distribution=mc_results['hed'].values,
            exposure_distributions=exposure_scenarios
        )
        
        # Combine all results
        assessment_results = {
            'hed_point_estimate': hed,
            'safety_assessment': safety,
            'monte_carlo_results': mc_results,
            'risk_assessment': risk_results,
            'sensitivity_analysis': sensitivity_results,
            'population_analysis': population_results
        }
        
        # Step 7: Generate reports and visualizations
        logger.info("Step 7: Generating reports and visualizations")
        
        # Save results
        results_path = output_dir / 'assessment_results.json'
        with open(results_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            def convert_for_json(obj):
                if isinstance(obj, (np.ndarray, np.generic)):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_for_json(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_for_json(item) for item in obj]
                else:
                    return obj
            
            json.dump(convert_for_json(assessment_results), f, indent=2)
        
        # Generate risk report
        risk_report = risk_assessor.generate_risk_report(assessment_results)
        with open(output_dir / 'risk_assessment_report.html', 'w') as f:
            f.write(risk_report)
        
        # Generate visualizations
        visualizer = RiskAssessmentVisualizer(output_dir)
        visualizer.create_all_figures(assessment_results)
        
        logger.info("Pipeline completed successfully!")
        logger.info(f"Results saved to: {output_dir}")
        
        # Print summary
        print("\n" + "="*60)
        print("RISK ASSESSMENT SUMMARY")
        print("="*60)
        print(f"Human Equivalent Dose (median): {risk_results['statistics']['median']:.4f} mg/kg/day")
        print(f"95% Confidence Interval: [{risk_results['statistics']['percentiles']['2.5']:.4f}, "
              f"{risk_results['statistics']['percentiles']['97.5']:.4f}] mg/kg/day")
        print(f"Probability of high risk (HED > 0.1): "
              f"{risk_results['risk_probabilities'].get('P(HED ≥ 0.1)', 0):.2%}")
        print(f"Margin of Safety (median): {safety['margin_of_safety']:.1f}x")
        print(f"Risk Level: {safety['risk_level']}")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    main()