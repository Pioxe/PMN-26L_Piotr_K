from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns


def split_dateset(df_clean, procent, random_state_new=67):
    # Separating features (X) from the target variable (y)
    X = df_clean.drop('target', axis=1)
    y = df_clean['target']

    # stratify=y is crucial: it maintains the original ratio of healthy vs sick patients in both sets, preventing bias during model training.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=procent, random_state=random_state_new, stratify=y)

    print("=== DATA BREAKDOWN SUMMARY ===")
    print(f"Entire collection: {len(X)} ")
    print(f"Training set: {len(X_train)} ")
    print(f"Test set (for model testing): {len(X_test)} ")

    print("\nChecking class balance (stratify):")
    print("Proportion sick patients in the training set: {:.1f}%".format((y_train.sum() / len(y_train)) * 100))
    print("Proportion sick patients in the test set: {:.1f}%".format((y_test.sum() / len(y_test)) * 100))
    return X_train, X_test, y_train, y_test

def run_tree_analysis(X_train, X_test, y_train, y_test, max_depth=None, random_state_new=67):
    # Initialize the model: if max_depth is provided, it limits the tree growth to prevent overfitting and improve model generalization.
    if max_depth is None:
        tree_model = DecisionTreeClassifier(random_state=random_state_new)
    else:
        tree_model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state_new)
    print(f"--- Training Model (max_depth={max_depth if max_depth else 'Unlimited'}) ---")
    
    tree_model.fit(X_train, y_train)

    y_pred = tree_model.predict(X_test)
    #Evaluation: Compare predictions with actual health status (y_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Healthy', 'Sick']))
    
    # Return both the trained model and its predictions for visualization functions
    return tree_model, y_pred 