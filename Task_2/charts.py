import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.tree import plot_tree
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

def plot_feature_vs_target(df, column_name):
    # custom colors for better visualization
    my_colors = {"Healthy": "#2ecc71", "Sick": "#e74c3c"} 
    temp_df = df.copy()
    temp_df['Health Condition'] = temp_df['target'].map({0: 'Healthy', 1: 'Sick'})
    
    unique_values = df[column_name].nunique()
    # If the feature has few unique values, treat it as categorical
    if unique_values < 10:
        sns.countplot(data=temp_df, x=column_name, hue='Health Condition', palette=my_colors)
    else:
        # For continuous data, histogram with a Kernel Density Estimate (KDE) curve
        
        sns.histplot(data=temp_df, x=column_name, hue='Health Condition', 
                     kde=True, palette='magma', common_norm=False)
    
    plt.title(f'"{column_name}" analysis')
    plt.xlabel(column_name)
    plt.ylabel('Patients')

def plot_tree_structure(tree_model, feature_names, class_names=['Healthy', 'Sick']):
    # depth of the tree to determine optimal visualization settings
    depth = tree_model.get_depth()
    # Dynamically adjusting figure size and font size to keep the plot readable
    # Larger trees require more space and smaller text to avoid node overlap
    fig_size = (20, 10) if depth <= 4 else (30, 15)
    font_size = 10 if depth <= 4 else 7

    plt.figure(figsize=fig_size, dpi=100)
    
    plot_tree(tree_model, 
              feature_names=list(feature_names), 
              class_names=class_names, 
              filled=True, 
              rounded=True, 
              impurity=True, 
              precision=2, 
              node_ids=True, 
              fontsize=font_size)
    
    plt.title(f"Decision Tree Structure (Depth: {depth})", fontsize=16, pad=20)
    plt.show()

def plot_feature_importance(tree_model, feature_names):
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': tree_model.feature_importances_
    }).sort_values(by='Importance', ascending=False) # Sort to show most influential features at the top

    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance_df, x='Importance', y='Feature', palette='magma')
    
    plt.title("Feature Importance", fontsize=14)
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    # grid on the X-axis for easier reading of precise values
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_test, y_pred):
    # Calculate the confusion matrix values
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    # annot=True displays the raw numbers, fmt='d' ensures they are formatted as integers
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Healthy', 'Sick'], 
                yticklabels=['Healthy', 'Sick'])
    plt.title('Confusion Matrix')
    plt.show()

def plot_roc_curve(tree_model, X_test, y_test):
    # predicted probabilities for the positive class (Sick)
    # Decision trees provide these based on the distribution of classes in the leaf nodes
    y_probs = tree_model.predict_proba(X_test)[:, 1]
    # Calculate False Positive Rate (FPR) and True Positive Rate (TPR)
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
    auc_score = roc_auc_score(y_test, y_probs)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', label=f'ROC curve (AUC = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.show()