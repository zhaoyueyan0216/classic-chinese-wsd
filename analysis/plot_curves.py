import csv
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

plt.rcParams['font.sans-serif'] = ['SimHei']  # for Chinese characters
plt.rcParams['axes.unicode_minus'] = False  # for minus signs

def safe_json_load(s):
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        s = s.strip()
        if s.startswith('[') and s.endswith(']'):
            try:
                content = s[1:-1]
                items = content.split(', ')
                items = [item.strip('"\'') for item in items]
                return items
            except:
                return []
        return []

def read_comparison_results(file_path):
    results = []
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            gold_wsid = row['Gold wsid']
            
            # Process Top-2 data
            run1_top2 = safe_json_load(row.get('Run1 Top-2', '[]'))
            run2_top2 = safe_json_load(row.get('Run2 Top-2', '[]'))
            run3_top2 = safe_json_load(row.get('Run3 Top-2', '[]'))
            
            # Check if gold is in each Top-2
            gold_in_run1 = gold_wsid in run1_top2
            gold_in_run2 = gold_wsid in run2_top2
            gold_in_run3 = gold_wsid in run3_top2
            
            # Calculate union
            union_wsids = set(run1_top2) | set(run2_top2) | set(run3_top2)
            gold_in_union = gold_wsid in union_wsids
            union_size = len(union_wsids)
            
            # Calculate how many times gold appears in the 3 Top-2 results
            recall_count = 0
            if gold_in_run1:
                recall_count += 1
            if gold_in_run2:
                recall_count += 1
            if gold_in_run3:
                recall_count += 1
            
            # Get correctness
            method_b_correct = row.get('Method B correct/incorrect', '').lower() == 'true'
            
            results.append({
                'gold_wsid': gold_wsid,
                'gold_in_union': gold_in_union,
                'union_size': union_size,
                'recall_count': recall_count,
                'method_b_correct': method_b_correct,
                'run1_top2': run1_top2,
                'run2_top2': run2_top2,
                'run3_top2': run3_top2
            })
    return results

def analyze_fixed_union_size(results):
    """Analyze relationship between recall count and accuracy for fixed union sizes"""
    
    union_size_data = {2: [], 3: [], 4: []}
    
    for result in results:
        union_size = result['union_size']
        if union_size in union_size_data:
            union_size_data[union_size].append({
                'recall_count': result['recall_count'],
                'correct': result['method_b_correct']
            })
    
    # Calculate accuracy for each recall count under each union size
    recall_accuracy_data = {2: {}, 3: {}, 4: {}}
    
    for union_size in [2, 3, 4]:
        # Group by recall count
        recall_groups = defaultdict(list)
        for item in union_size_data[union_size]:
            recall_groups[item['recall_count']].append(item['correct'])
        
        # Calculate accuracy for each recall count
        for recall_count, correct_list in recall_groups.items():
            if correct_list:
                accuracy = sum(correct_list) / len(correct_list)
                recall_accuracy_data[union_size][recall_count] = accuracy
    
    return recall_accuracy_data

def analyze_fixed_recall(results):
    """Analyze relationship between union size and accuracy for fixed recall counts"""
    
    # We fix recall count to different values and analyze the effect of union size
    recall_fixed_data = {1: [], 2: [], 3: []}
    
    for result in results:
        recall_count = result['recall_count']
        if recall_count in recall_fixed_data:
            recall_fixed_data[recall_count].append({
                'union_size': result['union_size'],
                'correct': result['method_b_correct']
            })
    
    # Calculate accuracy for each union size under each recall count
    union_size_accuracy_data = {1: {}, 2: {}, 3: {}}
    
    for recall_count in [1, 2, 3]:
        # Group by union size
        union_groups = defaultdict(list)
        for item in recall_fixed_data[recall_count]:
            union_groups[item['union_size']].append(item['correct'])
        
        # Calculate accuracy for each union size
        for union_size, correct_list in union_groups.items():
            if correct_list:
                accuracy = sum(correct_list) / len(correct_list)
                union_size_accuracy_data[recall_count][union_size] = accuracy
    
    return union_size_accuracy_data

def plot_fixed_union_size(recall_accuracy_data):
    """Plot Curve 1: Fixed union size, x=recall count, y=accuracy"""
    
    plt.figure(figsize=(10, 6))
    
    colors = ['blue', 'green', 'red']
    labels = ['Union Size=2', 'Union Size=3', 'Union Size=4']
    
    for i, union_size in enumerate([2, 3, 4]):
        data = recall_accuracy_data[union_size]
        if data:
            # Sort by recall count
            recall_counts = sorted(data.keys())
            accuracies = [data[rc] for rc in recall_counts]
            
            plt.plot(recall_counts, accuracies, marker='o', color=colors[i], label=labels[i])
    
    plt.xlabel('Recall Count (Number of times gold appears in 3 Top-2 runs)', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Curve 1: Fixed Union Size, Recall Count vs Accuracy', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.xticks([0, 1, 2, 3])
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig('curve1_fixed_union_size.png', dpi=300, bbox_inches='tight')
    print("Saved Curve 1: curve1_fixed_union_size.png")
    plt.show()

def plot_fixed_recall(union_size_accuracy_data):
    """Plot Curve 2: Fixed recall count, x=union size, y=accuracy"""
    
    plt.figure(figsize=(10, 6))
    
    colors = ['blue', 'green', 'red']
    labels = ['Recall Count=1', 'Recall Count=2', 'Recall Count=3']
    
    for i, recall_count in enumerate([1, 2, 3]):
        data = union_size_accuracy_data[recall_count]
        if data:
            # Sort by union size
            union_sizes = sorted(data.keys())
            accuracies = [data[us] for us in union_sizes]
            
            plt.plot(union_sizes, accuracies, marker='o', color=colors[i], label=labels[i])
    
    plt.xlabel('Union Size', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Curve 2: Fixed Recall Count, Union Size vs Accuracy', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.xticks([2, 3, 4, 5, 6])
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig('curve2_fixed_recall.png', dpi=300, bbox_inches='tight')
    print("Saved Curve 2: curve2_fixed_recall.png")
    plt.show()

def plot_additional_analysis(results):
    """Plot additional analysis graphs"""
    
    # Plot GoldInUnion vs Accuracy
    plt.figure(figsize=(10, 6))
    
    gold_in_union_correct = [r['method_b_correct'] for r in results if r['gold_in_union']]
    gold_not_in_union_correct = [r['method_b_correct'] for r in results if not r['gold_in_union']]
    
    categories = ['Gold In Union', 'Gold Not In Union']
    accuracies = [
        sum(gold_in_union_correct) / len(gold_in_union_correct) if gold_in_union_correct else 0,
        sum(gold_not_in_union_correct) / len(gold_not_in_union_correct) if gold_not_in_union_correct else 0
    ]
    
    bars = plt.bar(categories, accuracies, color=['green', 'red'], alpha=0.7)
    
    plt.xlabel('Gold Presence in Union', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('GoldInUnion vs Accuracy', fontsize=14)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height, f'{accuracies[i]:.2%}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('gold_in_union_vs_accuracy.png', dpi=300, bbox_inches='tight')
    print("Saved GoldInUnion comparison: gold_in_union_vs_accuracy.png")
    plt.show()

def main():
    file_path = 'comparison_results_20260418_191515.csv'
    results = read_comparison_results(file_path)
    
    print(f"Read {len(results)} samples")
    
    # Analyze fixed union size
    print("\nAnalyzing Curve 1: Fixed Union Size...")
    recall_accuracy_data = analyze_fixed_union_size(results)
    plot_fixed_union_size(recall_accuracy_data)
    
    # Analyze fixed recall count
    print("\nAnalyzing Curve 2: Fixed Recall Count...")
    union_size_accuracy_data = analyze_fixed_recall(results)
    plot_fixed_recall(union_size_accuracy_data)
    
    # Plot additional analysis
    print("\nPlotting additional analysis...")
    plot_additional_analysis(results)
    
    print("\nAll graphs saved successfully!")

if __name__ == "__main__":
    main()