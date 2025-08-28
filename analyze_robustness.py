import main
import pickle

# Load all  saved results
seeds_results = {}
dataset_name = "asset" #can be mpqa, sst5, trec and asset
optimization_mode = "random_wo_context" # random_vocab for mpqa, random_with_context for sst5,  random_wo_context for trec and asset
safe_model_name = "Qwen__Qwen2.5-0.5B"  #can be gpt2, EleutherAI__pythia-410m, Qwen__Qwen2.5-0.5B, EleutherAI__gpt-neo-125M, HuggingFaceTB__SmolLM2-360M
   
for seed in [1,2,3,4,5]:
    with open(f'robustness_results_{dataset_name}_{optimization_mode}_{seed}_{safe_model_name}.pkl', 'rb') as f:
        seeds_results[seed] = pickle.load(f)

# Get overall statistics
overall_stats = main.analyze_robustness_across_seeds(seeds_results)

def print_key_stats(overall_stats):
    
    print("\n" + "="*50)
    print("KEY ROBUSTNESS STATISTICS")
    print("="*50)
    
    print(f"Average Performance Drop:     {overall_stats['mean_avg_drop']:.3f} ± {overall_stats['std_avg_drop']:.3f}")
    print(f"Worst-Case Performance Drop:  {overall_stats['mean_worst_drop']:.3f} ± {overall_stats['std_worst_drop']:.3f}")
    print(f"Best-Case Performance Drop:   {overall_stats['mean_best_drop']:.3f} ± {overall_stats['std_best_drop']:.3f}")
    
    vulnerability_mode, vulnerability_count = overall_stats['most_common_vulnerability']
    print(f"Most Vulnerable to:           {vulnerability_mode} ({vulnerability_count} cases)")
    
    

print_key_stats(overall_stats)
