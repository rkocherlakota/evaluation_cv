from helper import calculate_metrics

# Example usage
ground_truth_file = './ground_truth.json'
predictions_file = './predictions.json'

keys = ['bbox', 'segmentation', 'category_id']
for key in keys:
    precision, recall, f1, jaccard = calculate_metrics(ground_truth_file, predictions_file, key)
    print(f'Metrics for {key}:')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1-Score: {f1:.4f}')
    print(f'IoU: {jaccard:.4f}\n')