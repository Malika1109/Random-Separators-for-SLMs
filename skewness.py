import numpy as np
from scipy import stats

# Script to plug in average drops fro a model-dataset pair and calculate skewness and kurtosis


avg_drops = [
    # Seed 1
    18.403, 16.058, 11.447, 2.231, 9.169,
    # Seed 2  
    12.965, 12.214, 3.377, 14.380, 13.988,
    # Seed 3
    15.124, 13.075, 11.324, 10.550, 14.481,
    # Seed 4
    17.759, 11.332, 16.881, 5.530, 15.570,
    # Seed 5
    9.838, 12.542, 1.511, 2.476, 0.769
]  
  
# # pythia MPQA Average Drop values (converted from percentages to decimals)
# avg_drops = [
#     # Seed 1
#     0.318, -0.012, 0.090, -0.023, 0.083,
#     # Seed 2  
#     0.005, 0.047, 0.092, 0.167, -0.021,
#     # Seed 3
#     0.033, 0.169, 0.080, 0.174, 0.082,
#     # Seed 4
#     0.114, 0.111, -0.002, -0.037, -0.018,
#     # Seed 5
#     0.078, 0.092, 0.066, -0.081, 0.049
# ]
 
 
# # GPT-2 MPQA Average Drop values (converted from percentages to decimals)
# avg_drops = [
#     # Seed 1
#     0.189, 0.232, 0.102, 0.073, 0.140,
#     # Seed 2  
#     0.045, -0.010, 0.006, 0.062, 0.084,
#     # Seed 3
#     0.075, 0.084, 0.086, 0.185, 0.034,
#     # Seed 4
#     0.100, -0.063, 0.127, 0.090, -0.074,
#     # Seed 5
#     0.030, 0.146, -0.033, 0.089, -0.040
# ]

# Calculate basic statistics
mean_drop = np.mean(avg_drops)
std_drop = np.std(avg_drops, ddof=1)  # Sample standard deviation
median_drop = np.median(avg_drops)

# Calculate skewness and kurtosis
skewness = stats.skew(avg_drops)
kurt = stats.kurtosis(avg_drops)  # Excess kurtosis (subtract 3 from Pearson kurtosis)

print("GPT-2 MPQA Separator Robustness Statistics")
print("=" * 50)
print(f"Number of separators: {len(avg_drops)}")
print(f"Mean drop: {mean_drop:.4f} ({mean_drop*100:.2f}%)")
print(f"Standard deviation: {std_drop:.4f} ({std_drop*100:.2f}%)")
print(f"Median drop: {median_drop:.4f} ({median_drop*100:.2f}%)")
print()
print(f"Skewness: {skewness:.3f}")
print(f"Kurtosis (excess): {kurt:.3f}")
print()

# Interpretation
print("Interpretation:")
print("-" * 20)
if skewness > 0.5:
    print(f"• POSITIVE SKEW ({skewness:.3f}): Most separators have low vulnerability,")
    print("  but a few separators are extremely vulnerable")
elif skewness < -0.5:
    print(f"• NEGATIVE SKEW ({skewness:.3f}): Most separators are vulnerable,")
    print("  but a few separators are exceptionally robust")
else:
    print(f"• APPROXIMATELY SYMMETRIC ({skewness:.3f}): Separator vulnerability")
    print("  is roughly normally distributed")

if kurt > 1:
    print(f"• HIGH KURTOSIS ({kurt:.3f}): Heavy tails - extreme vulnerability")
    print("  values occur more frequently than expected")
elif kurt < -1:
    print(f"• LOW KURTOSIS ({kurt:.3f}): Light tails - vulnerability values")
    print("  are more concentrated around the mean")
else:
    print(f"• NORMAL KURTOSIS ({kurt:.3f}): Tail behavior similar to normal distribution")

print()
print("Key Insights:")
print("• Negative drops indicate separators that IMPROVE with perturbation")
print(f"• {sum(1 for x in avg_drops if x < 0)} out of {len(avg_drops)} separators show improvement")
print(f"• Range: {min(avg_drops)*100:.1f}% to {max(avg_drops)*100:.1f}%")