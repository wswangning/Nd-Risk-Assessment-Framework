from src.analysis.risk_assessment import ProbabilisticRiskAssessment

# Initialize assessor
assessor = ProbabilisticRiskAssessment(config)

# Define parameter distributions
parameter_distributions = {
    'ec50': {
        'distribution': 'lognormal',
        'mean': np.log(50.0),
        'std': 0.3
    },
    'fu': {
        'distribution': 'normal',
        'mean': 0.176,
        'std': 0.05
    }
}

# Run Monte Carlo simulation
results = assessor.monte_carlo_simulation(
    model_function=my_hed_model,
    parameter_samples=parameter_samples
)

# Calculate risk probabilities
risk_probs = assessor.calculate_risk_probabilities(results['hed'])