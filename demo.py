from helper import calculate_metrics
import csv

ground_truth_file = './ground_truth.json'
prediction_files = ['./predictions1.json', './predictions2.json', './predictions3.json', './predictions4.json']

# Keys to calculate metrics for
keys = ['bbox', 'segmentation', 'category_id']

# CSV file to save results
output_csv_file = 'evaluation_results.csv'

# Open CSV file for writing
with open(output_csv_file, 'w', newline='') as csvfile:
    fieldnames = ['Prediction File', 'Metric', 'Precision', 'Recall', 'F1-Score', 'Jaccard Index']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    writer.writeheader()
    
    # Iterate over prediction files and keys
    for predictions_file in prediction_files:
        print(f'Processing file: {predictions_file}')
        for key in keys:
            precision, recall, f1, jaccard = calculate_metrics(ground_truth_file, predictions_file, key)
            print(f'Metrics for {key} from {predictions_file}: Precision: {precision}, Recall: {recall}, F1-Score: {f1}, Jaccard Index: {jaccard}')
            writer.writerow({
                'Prediction File': predictions_file,
                'Metric': key,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1,
                'Jaccard Index': jaccard
            })

print(f'Evaluation results saved to {output_csv_file}')