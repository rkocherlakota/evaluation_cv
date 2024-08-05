import json
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score

def calculate_metrics(ground_truth_file, predictions_file, key):
    with open(ground_truth_file, 'r') as f:
        ground_truth = json.load(f)
        
    with open(predictions_file, 'r') as f:
        predictions = json.load(f)
        
    # Create dictionaries for ground truth and predictions for quick lookup
    ground_truth_dict = {ann['id']: ann for ann in ground_truth['annotations']}
    predictions_dict = {ann['id']: ann for ann in predictions['annotations']}
    
    # Initialize lists to collect true and predicted values
    y_true = []
    y_pred = []
    
    # Extract the necessary data based on the key for matching annotations
    for ann_id in predictions_dict.keys():
        if ann_id in ground_truth_dict:
            gt = ground_truth_dict[ann_id]
            pred = predictions_dict[ann_id]
            
            if key == 'bbox':
                y_true.append(tuple(gt['bbox']))
                y_pred.append(tuple(pred['bbox']))
            elif key == 'segmentation':
                y_true.append(tuple(tuple(segment) for segment in gt['segmentation']))
                y_pred.append(tuple(tuple(segment) for segment in pred['segmentation']))
            elif key == 'category_id':
                y_true.append(gt['category_id'])
                y_pred.append(pred['category_id'])
            else:
                raise ValueError("Invalid key. Must be one of 'bbox', 'segmentation', 'category_id'")
    
    if key in ['bbox', 'segmentation']:
        mlb = MultiLabelBinarizer()
        y_true = mlb.fit_transform(y_true)
        y_pred = mlb.transform(y_pred)
    
    # Calculate metrics with zero_division set to avoid warnings
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    jaccard = jaccard_score(y_true, y_pred, average='weighted', zero_division=0)
    
    return precision, recall, f1, jaccard