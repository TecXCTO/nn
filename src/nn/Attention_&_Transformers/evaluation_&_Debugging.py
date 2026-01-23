⃣""" # Evaluation & Debugging
Metric	Use‑case
Accuracy	Classification
MSE / RMSE	Regression
BLEU / ROUGE	NLP generation
Confusion Matrix	Error analysis
Grad‑CAM	Visualise CNN focus
"""
from sklearn.metrics import accuracy_score, confusion_matrix

preds = model(X_test).argmax(dim=1).cpu().numpy()
print("Accuracy:", accuracy_score(y_test, preds))
print("Confusion Matrix:\n", confusion_matrix(y_test, preds))