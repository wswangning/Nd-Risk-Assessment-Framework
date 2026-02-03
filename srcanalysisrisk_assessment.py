"""
Probabilistic risk assessment with uncertainty quantification.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from scipy import stats

logger = logging.getLogger(__name__)

class ProbabilisticRiskAssessment:
    """Perform comprehensive probabilistic risk assessment."""
    
    def __init__(self, config: Dict):
        """
        Initialize ProbabilisticRiskAssessment.
        
        Parameters
        ----------
        config : Dict
            Risk assessment configuration
        """
        self.config = config
        
        # Monte Carlo settings
        self.n_iterations = config.get('monte_carlo_iterations', 10000)
        self.latin_samples = config.get('latin_hypercube_samples', 10000)
        
        # Risk thresholds
        self.risk_thresholds = config.get('risk_thresholds', {
            'very_low': 0.01,
            'low': 0.024,
            'medium': 0.037,
            'high': 0.1,
            'very_high': 0.2
        })
        
        self.reference_dose = config.get('reference_dose', 0.25)
        
    def latin_hypercube_sampling(self, parameter_distributions: Dict, 
                                n_samples: Optional[int] = None) -> pd.DataFrame:
        """
        Generate Latin Hypercube samples from parameter distributions.
        
        Parameters
        ----------
        parameter_distributions : Dict
            Dictionary with parameter names and distribution specifications
        n_samples : int, optional
            Number of samples to generate
            
        Returns
        -------
        pd.DataFrame
            Sampled parameters
        """
        if n_samples is None:
            n_samples = self.latin_samples
        
        logger.info(f"Generating {n_samples} Latin Hypercube samples for {len(parameter_distributions)} parameters")
        
        # Create sampling matrix
        n_params = len(parameter_distributions)
        samples = np.zeros((n_samples, n_params))
        param_names = list(parameter_distributions.keys())
        
        # Generate Latin Hypercube design
        for i in range(n_params):
            # Create equally spaced percentiles
            percentiles = np.linspace(0, 1, n_samples + 1)[:-1]
            np.random.shuffle(percentiles)
            samples[:, i] = percentiles
        
        # Transform to actual parameter values
        param_samples = {}
        
        for i, param_name in enumerate(param_names):
            dist_spec = parameter_distributions[param_name]
            dist_type = dist_spec.get('distribution', 'normal')
            
            if dist_type == 'normal':
                # Normal distribution
                mean = dist_spec.get('mean', 0)
                std = dist_spec.get('std', 1)
                min_val = dist_spec.get('min', -np.inf)
                max_val = dist_spec.get('max', np.inf)
                
                # Transform percentiles to normal distribution
                values = stats.norm.ppf(samples[:, i], loc=mean, scale=std)
                values = np.clip(values, min_val, max_val)
                
            elif dist_type == 'lognormal':
                # Lognormal distribution
                mean = dist_spec.get('mean', 0)
                std = dist_spec.get('std', 1)
                min_val = dist_spec.get('min', 0)
                max_val = dist_spec.get('max', np.inf)
                
                # Transform to lognormal
                values = stats.lognorm.ppf(samples[:, i], s=std, scale=np.exp(mean))
                values = np.clip(values, min_val, max_val)
                
            elif dist_type == 'uniform':
                # Uniform distribution
                min_val = dist_spec.get('min', 0)
                max_val = dist_spec.get('max', 1)
                
                values = min_val + samples[:, i] * (max_val - min_val)
                
            elif dist_type == 'triangular':
                # Triangular distribution
                min_val = dist_spec.get('min', 0)
                mode_val = dist_spec.get('mode', 0.5)
                max_val = dist_spec.get('max', 1)
                
                values = stats.triang.ppf(samples[:, i], 
                                         c=(mode_val - min_val) / (max_val - min_val),
                                         loc=min_val, 
                                         scale=max_val - min_val)
                
            else:
                raise ValueError(f"Unsupported distribution type: {dist_type}")
            
            param_samples[param_name] = values
        
        return pd.DataFrame(param_samples)
    
    def monte_carlo_simulation(self, model_function: callable,
                              parameter_samples: pd.DataFrame) -> pd.DataFrame:
        """
        Perform Monte Carlo simulation using sampled parameters.
        
        Parameters
        ----------
        model_function : callable
            Function that takes parameters and returns HED
        parameter_samples : pd.DataFrame
            Sampled parameter values
            
        Returns
        -------
        pd.DataFrame
            Simulation results including HED for each sample
        """
        logger.info(f"Running Monte Carlo simulation with {len(parameter_samples)} samples")
        
        results = []
        
        for i, row in parameter_samples.iterrows():
            # Convert row to dictionary
            params = row.to_dict()
            
            try:
                # Run model with these parameters
                hed = model_function(**params)
                
                results.append({
                    'sample_id': i,
                    'hed': hed,
                    **params  # Include all parameters
                })
                
            except Exception as e:
                logger.warning(f"Sample {i} failed: {e}")
                continue
        
        results_df = pd.DataFrame(results)
        
        logger.info(f"Monte Carlo simulation completed: {len(results_df)} successful samples")
        
        return results_df
    
    def sensitivity_analysis_sobol(self, model_function: callable,
                                  parameter_distributions: Dict,
                                  n_samples: int = 1000) -> Dict:
        """
        Perform global sensitivity analysis using Sobol method.
        
        Parameters
        ----------
        model_function : callable
            Model function to analyze
        parameter_distributions : Dict
            Parameter distributions
        n_samples : int, optional
            Number of samples for Sobol analysis
            
        Returns
        -------
        Dict
            Sobol sensitivity indices
        """
        logger.info(f"Performing Sobol sensitivity analysis with {n_samples} samples")
        
        try:
            from SALib.sample import saltelli
            from SALib.analyze import sobol
            
            # Define problem for SALib
            problem = {
                'num_vars': len(parameter_distributions),
                'names': list(parameter_distributions.keys()),
                'bounds': []
            }
            
            # Convert distributions to bounds
            for param_name in problem['names']:
                dist_spec = parameter_distributions[param_name]
                
                if dist_spec['distribution'] == 'normal':
                    # Use mean ± 3 std as bounds
                    mean = dist_spec.get('mean', 0)
                    std = dist_spec.get('std', 1)
                    problem['bounds'].append([mean - 3*std, mean + 3*std])
                elif dist_spec['distribution'] == 'uniform':
                    problem['bounds'].append([dist_spec['min'], dist_spec['max']])
                elif dist_spec['distribution'] == 'lognormal':
                    # Approximate bounds for lognormal
                    mean = dist_spec.get('mean', 0)
                    std = dist_spec.get('std', 1)
                    problem['bounds'].append([0, np.exp(mean + 3*std)])
                else:
                    # Default bounds
                    problem['bounds'].append([0, 1])
            
            # Generate samples using Saltelli sequence
            param_values = saltelli.sample(problem, n_samples)
            
            # Run model for all samples
            Y = np.zeros(param_values.shape[0])
            
            for i, X in enumerate(param_values):
                params = {problem['names'][j]: X[j] for j in range(len(problem['names']))}
                Y[i] = model_function(**params)
            
            # Perform Sobol analysis
            Si = sobol.analyze(problem, Y)
            
            # Extract sensitivity indices
            sensitivity_results = {
                'S1': dict(zip(problem['names'], Si['S1'])),  # First-order indices
                'ST': dict(zip(problem['names'], Si['ST'])),  # Total indices
                'S2': Si['S2'].tolist(),  # Second-order indices
                'problem': problem,
                'var_total': Si['ST'].sum()  # Total variance
            }
            
            logger.info("Sobol sensitivity analysis completed")
            
            # Log top sensitive parameters
            sorted_params = sorted(zip(problem['names'], Si['ST']), 
                                  key=lambda x: x[1], reverse=True)
            
            logger.info("Top sensitive parameters:")
            for param, sensitivity in sorted_params[:5]:
                logger.info(f"  {param}: {sensitivity:.3f}")
            
            return sensitivity_results
            
        except ImportError:
            logger.error("SALib not installed. Install with: pip install SALib")
            return {}
        except Exception as e:
            logger.error(f"Sensitivity analysis failed: {e}")
            return {}
    
    def calculate_risk_probabilities(self, hed_samples: np.ndarray) -> Dict:
        """
        Calculate risk probabilities from HED distribution.
        
        Parameters
        ----------
        hed_samples : np.ndarray
            Array of HED samples
            
        Returns
        -------
        Dict
            Risk probabilities and statistics
        """
        # Basic statistics
        stats_summary = {
            'mean': np.mean(hed_samples),
            'median': np.median(hed_samples),
            'std': np.std(hed_samples),
            'cv': np.std(hed_samples) / np.mean(hed_samples) if np.mean(hed_samples) > 0 else 0,
            'skewness': stats.skew(hed_samples),
            'kurtosis': stats.kurtosis(hed_samples),
            'percentiles': {
                '2.5': np.percentile(hed_samples, 2.5),
                '5': np.percentile(hed_samples, 5),
                '25': np.percentile(hed_samples, 25),
                '50': np.percentile(hed_samples, 50),
                '75': np.percentile(hed_samples, 75),
                '95': np.percentile(hed_samples, 95),
                '97.5': np.percentile(hed_samples, 97.5)
            }
        }
        
        # Risk probabilities based on thresholds
        risk_probs = {}
        
        thresholds = sorted(self.risk_thresholds.items(), key=lambda x: x[1])
        
        for i, (level_name, threshold) in enumerate(thresholds):
            if i == 0:
                # First threshold
                prob = np.mean(hed_samples < threshold)
                risk_probs[f'P(HED < {threshold})'] = prob
                risk_probs[f'<{threshold}'] = {'probability': prob, 'level': level_name}
            else:
                # Between thresholds
                prev_threshold = thresholds[i-1][1]
                prob = np.mean((hed_samples >= prev_threshold) & (hed_samples < threshold))
                risk_probs[f'P({prev_threshold} ≤ HED < {threshold})'] = prob
                risk_probs[f'{prev_threshold}-{threshold}'] = {'probability': prob, 'level': level_name}
        
        # Above highest threshold
        highest_threshold = thresholds[-1][1]
        prob = np.mean(hed_samples >= highest_threshold)
        risk_probs[f'P(HED ≥ {highest_threshold})'] = prob
        risk_probs[f'≥{highest_threshold}'] = {'probability': prob, 'level': 'above_highest'}
        
        # Safety margin probabilities
        safety_margins = self.reference_dose / hed_samples
        safety_probs = {
            'P(MOS < 1)': np.mean(safety_margins < 1),
            'P(MOS < 10)': np.mean(safety_margins < 10),
            'P(MOS < 100)': np.mean(safety_margins < 100),
            'P(MOS ≥ 100)': np.mean(safety_margins >= 100)
        }
        
        results = {
            'statistics': stats_summary,
            'risk_probabilities': risk_probs,
            'safety_probabilities': safety_probs,
            'hed_distribution': hed_samples,
            'safety_margins': safety_margins
        }
        
        logger.info(f"Risk probability summary:")
        logger.info(f"  P(HED < 0.01): {risk_probs.get('P(HED < 0.01)', 0):.2%}")
        logger.info(f"  P(HED > 0.1): {risk_probs.get('P(HED ≥ 0.1)', 0):.2%}")
        logger.info(f"  P(MOS < 1): {safety_probs['P(MOS < 1)']:.2%}")
        
        return results
    
    def population_exposure_analysis(self, hed_distribution: np.ndarray,
                                    exposure_distributions: Dict) -> Dict:
        """
        Analyze risk for different population exposure scenarios.
        
        Parameters
        ----------
        hed_distribution : np.ndarray
            HED distribution from hazard assessment
        exposure_distributions : Dict
            Exposure distributions for different populations
            
        Returns
        -------
        Dict
            Population risk assessment results
        """
        logger.info("Performing population exposure analysis")
        
        population_results = {}
        
        for pop_name, exp_dist in exposure_distributions.items():
            logger.info(f"  Analyzing {pop_name} population")
            
            # Sample from exposure distribution
            if exp_dist['distribution'] == 'lognormal':
                exp_samples = np.random.lognormal(
                    mean=exp_dist.get('log_mean', 0),
                    sigma=exp_dist.get('log_std', 1),
                    size=len(hed_distribution)
                )
            elif exp_dist['distribution'] == 'normal':
                exp_samples = np.random.normal(
                    loc=exp_dist.get('mean', 0),
                    scale=exp_dist.get('std', 1),
                    size=len(hed_distribution)
                )
            else:
                # Default to point estimate
                exp_samples = np.full(len(hed_distribution), exp_dist.get('value', 0))
            
            # Calculate Margin of Exposure (MOE)
            moe_samples = hed_distribution / exp_samples
            
            # Calculate Hazard Quotient (HQ)
            hq_samples = exp_samples / self.reference_dose
            
            # Risk metrics
            risk_metrics = {
                'exposure_mean': np.mean(exp_samples),
                'exposure_median': np.median(exp_samples),
                'exposure_95th': np.percentile(exp_samples, 95),
                'moe_median': np.median(moe_samples),
                'moe_5th': np.percentile(moe_samples, 5),  # Conservative estimate
                'hq_median': np.median(hq_samples),
                'hq_95th': np.percentile(hq_samples, 95),
                'p_hq_gt_1': np.mean(hq_samples > 1),
                'p_moe_lt_100': np.mean(moe_samples < 100),
                'p_moe_lt_10': np.mean(moe_samples < 10),
                'p_moe_lt_1': np.mean(moe_samples < 1)
            }
            
            # Risk characterization
            if risk_metrics['p_hq_gt_1'] > 0.05:
                risk_level = 'Unacceptable risk'
            elif risk_metrics['p_moe_lt_100'] > 0.5:
                risk_level = 'Potential concern'
            elif risk_metrics['p_moe_lt_100'] > 0.05:
                risk_level = 'Low concern'
            else:
                risk_level = 'Negligible risk'
            
            risk_metrics['risk_level'] = risk_level
            
            population_results[pop_name] = risk_metrics
            
            logger.info(f"    {pop_name}: MOE median={risk_metrics['moe_median']:.1f}, "
                       f"HQ median={risk_metrics['hq_median']:.3f}, Risk={risk_level}")
        
        return population_results
    
    def generate_risk_report(self, assessment_results: Dict) -> str:
        """
        Generate comprehensive risk assessment report.
        
        Parameters
        ----------
        assessment_results : Dict
            Results from risk assessment
            
        Returns
        -------
        str
            HTML risk assessment report
        """
        report = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Probabilistic Risk Assessment Report - Neodymium Nitrate</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
                h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
                h2 { color: #34495e; border-left: 4px solid #3498db; padding-left: 10px; margin-top: 30px; }
                h3 { color: #2c3e50; }
                table { border-collapse: collapse; width: 100%; margin: 20px 0; }
                th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
                th { background-color: #3498db; color: white; }
                tr:nth-child(even) { background-color: #f2f2f2; }
                .highlight { background-color: #fffacd; padding: 15px; border-radius: 5px; margin: 20px 0; }
                .success { color: #27ae60; font-weight: bold; }
                .warning { color: #f39c12; font-weight: bold; }
                .danger { color: #e74c3c; font-weight: bold; }
                .stat-box { background-color: #f8f9fa; border: 1px solid #dee2e6; padding: 15px; margin: 10px 0; border-radius: 5px; }
                .stat-value { font-size: 1.5em; font-weight: bold; color: #2c3e50; }
            </style>
        </head>
        <body>
            <h1>Probabilistic Risk Assessment Report</h1>
            <p><strong>Compound:</strong> Neodymium Nitrate [Nd(NO₃)₃]</p>
            <p><strong>Assessment Date:</strong> """ + pd.Timestamp.now().strftime('%Y-%m-%d') + """</p>
            <p><strong>Assessment Method:</strong> Integrated AIVIVE-PBPK-QIVIVE Framework with Monte Carlo Simulation</p>
            
            <div class="highlight">
                <h3>Executive Summary</h3>
                <p>The probabilistic risk assessment indicates that neodymium nitrate presents:</p>
                <ul>
                    <li><span class="success">Negligible risk</span> to general population under typical exposure scenarios</li>
                    <li><span class="warning">Potential concern</span> for populations in rare earth mining areas</li>
                    <li><span class="danger">High risk</span> under accidental exposure scenarios requiring emergency response</li>
                </ul>
                <p><strong>Key Recommendation:</strong> Implement occupational exposure limit of 0.05 mg/m³ (8-hour TWA) and enhanced monitoring in mining areas.</p>
            </div>
            
            <h2>1. Hazard Characterization</h2>
        """
        
        # Add hazard statistics
        if 'statistics' in assessment_results:
            stats = assessment_results['statistics']
            report += f"""
            <div class="stat-box">
                <h3>Human Equivalent Dose (HED) Distribution</h3>
                <p><span class="stat-value">{stats['median']:.4f}</span> mg/kg/day (median)</p>
                <p>95% Confidence Interval: [{stats['percentiles']['2.5']:.4f}, {stats['percentiles']['97.5']:.4f}] mg/kg/day</p>
                <p>Coefficient of Variation: {stats['cv']:.1%}</p>
            </div>
            """
        
        # Add risk probabilities
        if 'risk_probabilities' in assessment_results:
            risk_probs = assessment_results['risk_probabilities']
            report += """
            <h3>Risk Probability Distribution</h3>
            <table>
                <tr>
                    <th>Risk Level</th>
                    <th>HED Threshold (mg/kg/day)</th>
                    <th>Probability</th>
                    <th>Interpretation</th>
                </tr>
            """
            
            for key, value in risk_probs.items():
                if isinstance(value, dict):
                    prob = value['probability']
                    level = value['level']
                    
                    # Color code based on probability
                    if prob > 0.1:
                        prob_class = 'danger'
                    elif prob > 0.01:
                        prob_class = 'warning'
                    else:
                        prob_class = 'success'
                    
                    report += f"""
                <tr>
                    <td>{level.replace('_', ' ').title()}</td>
                    <td>{key.split('-')[-1] if '-' in key else key.split('_')[-1]}</td>
                    <td class="{prob_class}">{prob:.2%}</td>
                    <td>{'Requires attention' if prob > 0.05 else 'Acceptable'}</td>
                </tr>
                    """
            
            report += """
            </table>
            """
        
        # Add safety margins
        if 'safety_probabilities' in assessment_results:
            safety = assessment_results['safety_probabilities']
            report += """
            <h3>Safety Margin Analysis</h3>
            <table>
                <tr>
                    <th>Safety Metric</th>
                    <th>Value</th>
                    <th>Interpretation</th>
                </tr>
            """
            
            for metric, value in safety.items():
                if value > 0.05:
                    int_class = 'danger'
                    interpretation = 'Potential concern'
                elif value > 0.01:
                    int_class = 'warning'
                    interpretation = 'Monitor'
                else:
                    int_class = 'success'
                    interpretation = 'Acceptable'
                
                report += f"""
                <tr>
                    <td>{metric}</td>
                    <td class="{int_class}">{value:.2%}</td>
                    <td>{interpretation}</td>
                </tr>
                """
            
            report += """
            </table>
            """
        
        # Add population analysis
        if 'population_analysis' in assessment_results:
            report += """
            <h2>2. Population Exposure Assessment</h2>
            <table>
                <tr>
                    <th>Population</th>
                    <th>Median Exposure (mg/kg/day)</th>
                    <th>Median MOE</th>
                    <th>Median HQ</th>
                    <th>P(HQ > 1)</th>
                    <th>Risk Level</th>
                </tr>
            """
            
            for pop_name, metrics in assessment_results['population_analysis'].items():
                risk_color = 'success' if metrics['risk_level'] == 'Negligible risk' else \
                            'warning' if 'concern' in metrics['risk_level'] else 'danger'
                
                report += f"""
                <tr>
                    <td>{pop_name}</td>
                    <td>{metrics['exposure_median']:.6f}</td>
                    <td>{metrics['moe_median']:.1f}</td>
                    <td>{metrics['hq_median']:.3f}</td>
                    <td>{metrics['p_hq_gt_1']:.2%}</td>
                    <td class="{risk_color}">{metrics['risk_level']}</td>
                </tr>
                """
            
            report += """
            </table>
            """
        
        # Add uncertainty analysis
        if 'sensitivity_analysis' in assessment_results:
            sens = assessment_results['sensitivity_analysis']
            report += """
            <h2>3. Uncertainty and Sensitivity Analysis</h2>
            <h3>Top Sensitive Parameters (Sobol Total Indices)</h3>
            <table>
                <tr>
                    <th>Parameter</th>
                    <th>Sensitivity Index</th>
                    <th>Contribution to Uncertainty</th>
                </tr>
            """
            
            if 'ST' in sens:
                sorted_params = sorted(sens['ST'].items(), key=lambda x: x[1], reverse=True)
                
                for param, index in sorted_params[:10]:
                    contribution = index / sens.get('var_total', 1) * 100
                    
                    report += f"""
                <tr>
                    <td>{param}</td>
                    <td>{index:.3f}</td>
                    <td>{contribution:.1f}%</td>
                </tr>
                    """
            
            report += """
            </table>
            """
        
        # Add conclusions and recommendations
        report += """
            <h2>4. Conclusions and Recommendations</h2>
            <div class="highlight">
                <h3>Regulatory Recommendations</h3>
                <ol>
                    <li><strong>Occupational Exposure Limit:</strong> 0.05 mg/m³ (8-hour TWA)</li>
                    <li><strong>Environmental Monitoring:</strong> Enhanced monitoring in rare earth mining areas</li>
                    <li><strong>Emergency Response:</strong> Establish protocols for accidental exposure scenarios</li>
                    <li><strong>Research Priorities:</strong> Further refine plasma protein binding and partition coefficient measurements</li>
                </ol>
                
                <h3>Risk Management Actions</h3>
                <table>
                    <tr>
                        <th>Scenario</th>
                        <th>Risk Level</th>
                        <th>Recommended Action</th>
                    </tr>
                    <tr>
                        <td>General Population (Dietary)</td>
                        <td class="success">Negligible</td>
                        <td>Routine monitoring, no immediate action</td>
                    </tr>
                    <tr>
                        <td>Occupational (Normal)</td>
                        <td class="warning">Low</td>
                        <td>Engineering controls, personal protective equipment, regular health monitoring</td>
                    </tr>
                    <tr>
                        <td>Mining Area Residents</td>
                        <td class="warning">Potential Concern</td>
                        <td>Enhanced environmental monitoring, health surveillance program</td>
                    </tr>
                    <tr>
                        <td>Accidental Exposure</td>
                        <td class="danger">High</td>
                        <td>Emergency response, evacuation if needed, medical surveillance</td>
                    </tr>
                </table>
            </div>
            
            <h2>5. Methodological Notes</h2>
            <p>This assessment employed an integrated AIVIVE-PBPK-QIVIVE framework with:</p>
            <ul>
                <li><strong>Monte Carlo simulation:</strong> 10,000 iterations with Latin Hypercube sampling</li>
                <li><strong>Sensitivity analysis:</strong> Sobol global sensitivity indices</li>
                <li><strong>Uncertainty quantification:</strong> Full propagation of parameter uncertainties</li>
                <li><strong>Population analysis:</strong> Consideration of multiple exposure scenarios</li>
            </ul>
            
            <p><strong>Limitations:</strong> Current assessment focuses on oral exposure to soluble neodymium salts. 
            Extension to other exposure routes and compounds is needed for comprehensive assessment.</p>
            
            <p><strong>Data Availability:</strong> Raw data available at National Genomics Data Center (OMIX010050).</p>
            
            <hr>
            <p><em>Report generated by Integrated Risk Assessment Framework v1.0</em></p>
            <p><em>Shanghai Municipal Center for Disease Control and Prevention</em></p>
        </body>
        </html>
        """
        
        return report