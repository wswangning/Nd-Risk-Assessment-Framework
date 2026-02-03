from src.models.pbpk_model import NonlinearPBPKModel

# Initialize PBPK model
pbpk = NonlinearPBPKModel(config)

# Simulate oral exposure
results = pbpk.simulate(
    dose=10.0,      # mg/kg
    route='oral',
    duration=72.0,  # hours
    time_points=100
)

# Extract liver concentration profile
liver_concentration = results['concentrations']['liver']
time = results['time']