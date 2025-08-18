import trial_main
import pickle

# Load all  saved results
seeds_results = {}
dataset_name = "asset"
optimization_mode = "random_wo_context" 
safe_model_name = "Qwen__Qwen2.5-0.5B"

for seed in [1,2,3,4,5]:
    with open(f'robustness_results_{dataset_name}_{optimization_mode}_{seed}_{safe_model_name}.pkl', 'rb') as f:
        seeds_results[seed] = pickle.load(f)

# Get overall statistics
overall_stats = trial_main.analyze_robustness_across_seeds(seeds_results)

def print_key_stats(overall_stats):
    
    print("\n" + "="*50)
    print("KEY ROBUSTNESS STATISTICS")
    print("="*50)
    
    print(f"Average Performance Drop:     {overall_stats['mean_avg_drop']:.3f} ± {overall_stats['std_avg_drop']:.3f}")
    print(f"Worst-Case Performance Drop:  {overall_stats['mean_worst_drop']:.3f} ± {overall_stats['std_worst_drop']:.3f}")
    print(f"Best-Case Performance Drop:   {overall_stats['mean_best_drop']:.3f} ± {overall_stats['std_best_drop']:.3f}")
    
    vulnerability_mode, vulnerability_count = overall_stats['most_common_vulnerability']
    print(f"Most Vulnerable to:           {vulnerability_mode} ({vulnerability_count} cases)")
    
    print("\n" + "="*50)
    print("INTERPRETATION")
    print("="*50)
    
    # Convert to percentages for easier interpretation
    avg_pct = overall_stats['mean_avg_drop'] * 100
    worst_pct = overall_stats['mean_worst_drop'] * 100
    
    print(f"• Separators lose {avg_pct:.1f}% performance on average when perturbed")
    print(f"• In worst cases, performance drops by {worst_pct:.1f}%")
    print(f"• {vulnerability_mode} perturbations are most damaging")
    
    # Robustness assessment
    if overall_stats['mean_avg_drop'] < 0.05:
        robustness_level = "HIGHLY ROBUST"
    elif overall_stats['mean_avg_drop'] < 0.10:
        robustness_level = "MODERATELY ROBUST"
    elif overall_stats['mean_avg_drop'] < 0.20:
        robustness_level = "MODERATELY VULNERABLE"
    else:
        robustness_level = "HIGHLY VULNERABLE"
    
    print(f"• Overall Assessment: {robustness_level}")

print_key_stats(overall_stats)
