from src.models.qivive_calculations import QIVIVECalculator

# Initialize calculator
calculator = QIVIVECalculator(config)

# Calculate biophase concentration
c_bio = calculator.calculate_biophase_concentration(
    ec50=50.0,    # μM
    fu=0.176,     # Free fraction
    k_cell=1.2    # Partition coefficient
)

# Calculate human equivalent dose
hed = calculator.calculate_human_equivalent_dose(c_bio)

# Calculate safety margin
safety = calculator.calculate_safety_margin(hed)
print(f"HED: {hed:.4f} mg/kg/day")
print(f"Margin of Safety: {safety['margin_of_safety']:.1f}x")