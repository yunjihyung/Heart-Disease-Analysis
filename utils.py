# ✅ Accuracy, Precision, Recall, F1-score (전체 지표)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_auc_score

def evaluate_scores(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"ROC AUC   : {roc_auc_score(y_true, y_pred):.4f}")
