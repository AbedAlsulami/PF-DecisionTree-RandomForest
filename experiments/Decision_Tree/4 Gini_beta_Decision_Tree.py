# breast_cancer_gini_beta_tree_no_feature_selection.py
# Gini Beta Decision Tree without feature selection

import pandas as pd
import numpy as np
from itertools import product
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_curve, auc
from numpy import mean
from scipy.stats import sem
import matplotlib.pyplot as plt
import xlsxwriter


# ----------------------------
# Utility Functions
# ----------------------------

def preprocess_train_validation(X_train, X_validation):
    """Fit preprocessing on the training fold, then transform both folds."""
    X_train = pd.DataFrame(X_train).copy()
    X_validation = pd.DataFrame(X_validation).copy()

    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [col for col in X_train.columns if col not in numeric_cols]

    if numeric_cols:
        numeric_imputer = SimpleImputer(strategy="mean")
        X_train[numeric_cols] = numeric_imputer.fit_transform(X_train[numeric_cols])
        X_validation[numeric_cols] = numeric_imputer.transform(X_validation[numeric_cols])

    if categorical_cols:
        categorical_imputer = SimpleImputer(strategy="most_frequent")
        X_train[categorical_cols] = categorical_imputer.fit_transform(X_train[categorical_cols])
        X_validation[categorical_cols] = categorical_imputer.transform(X_validation[categorical_cols])

        for column in categorical_cols:
            encoder = LabelEncoder()
            train_values = X_train[column].astype(str)
            encoder.fit(train_values)

            mapping = {label: index for index, label in enumerate(encoder.classes_)}
            X_train[column] = train_values.map(mapping).astype(int)
            X_validation[column] = (
                X_validation[column].astype(str).map(mapping).fillna(-1).astype(int)
            )

    return X_train, X_validation


def evaluate_params(X_train, y_train, X_validation, y_validation,
                    beta, random_state):

    model = DecisionTreeClassifier(
        criterion="gini_beta",
        beta=beta,
        random_state=random_state
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_validation)

    return {
        "beta": beta,
        "score": accuracy_score(y_validation, predictions)
    }


# ----------------------------
# Repeated Experiment
# ----------------------------

grid_values = [0.01, 0.1, 0.2, 0.25, 0.3, 0.35, 0.45, 0.5]
grid_params = [(beta,) for beta in grid_values]

all_repeats = []

for repeat in range(1, 101):
    print(f"\n🔁 Repeat {repeat} of 100")

    data = pd.read_csv("Breast_Cancer.csv")
    df = pd.DataFrame(data)

    # Exclude Status from predictors explicitly.
    features = df.drop(columns=["Status"])
    y = df["Status"]

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        features,
        y,
        test_size=0.2,
        stratify=y,
        random_state=repeat
    )

    outer_cv = StratifiedKFold(
        n_splits=10,
        shuffle=True,
        random_state=repeat
    )

    nested_metrics = []

    for fold_number, (train_index, validation_index) in enumerate(
        outer_cv.split(X_train_full, y_train_full), start=1
    ):
        X_train_fold = X_train_full.iloc[train_index]
        X_validation_fold = X_train_full.iloc[validation_index]
        y_train_fold = y_train_full.iloc[train_index]
        y_validation_fold = y_train_full.iloc[validation_index]

        X_train_fold, X_validation_fold = preprocess_train_validation(
            X_train_fold,
            X_validation_fold
        )

        grid_results = Parallel(n_jobs=-1)(
            delayed(evaluate_params)(
                X_train_fold,
                y_train_fold,
                X_validation_fold,
                y_validation_fold,
                beta,
                repeat
            )
            for (beta,) in grid_params
        )

        best_result = max(grid_results, key=lambda result: result["score"])

        final_model = DecisionTreeClassifier(
            criterion="gini_beta",
            beta=best_result["beta"],
            random_state=repeat
        )

        # All columns are used; there is no feature-selection stage.
        final_model.fit(X_train_fold, y_train_fold)

        y_validation_pred = final_model.predict(X_validation_fold)

        alive_index = list(final_model.classes_).index("Alive")
        y_validation_prob = final_model.predict_proba(X_validation_fold)[:, alive_index]

        accuracy = accuracy_score(y_validation_fold, y_validation_pred)
        sensitivity = recall_score(
            y_validation_fold,
            y_validation_pred,
            pos_label="Alive",
            zero_division=0
        )
        specificity = recall_score(
            y_validation_fold,
            y_validation_pred,
            pos_label="Dead",
            zero_division=0
        )
        f1 = f1_score(
            y_validation_fold,
            y_validation_pred,
            pos_label="Alive",
            zero_division=0
        )
        fpr, tpr, _ = roc_curve(
            y_validation_fold,
            y_validation_prob,
            pos_label="Alive"
        )
        auc_value = auc(fpr, tpr)

        print(
            f"  Fold {fold_number:02d} | "
            f"beta={best_result['beta']} | "
            f"Depth={final_model.get_depth()} | "
            f"Leaves={final_model.get_n_leaves()}"
        )


        nested_metrics.append({
            "accuracy": accuracy,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "f1": f1,
            "auc": auc_value,
            "Depth": final_model.get_depth(),
            "Leaves": final_model.get_n_leaves(),
            "beta": best_result["beta"]
        })

    mean_acc = mean([metric["accuracy"] for metric in nested_metrics])
    se_acc = sem([metric["accuracy"] for metric in nested_metrics])

    mean_sens = mean([metric["sensitivity"] for metric in nested_metrics])
    se_sens = sem([metric["sensitivity"] for metric in nested_metrics])

    mean_spec = mean([metric["specificity"] for metric in nested_metrics])
    se_spec = sem([metric["specificity"] for metric in nested_metrics])

    mean_f1 = mean([metric["f1"] for metric in nested_metrics])
    se_f1 = sem([metric["f1"] for metric in nested_metrics])

    mean_auc = mean([metric["auc"] for metric in nested_metrics])
    se_auc = sem([metric["auc"] for metric in nested_metrics])

    mean_depth = mean([metric["Depth"] for metric in nested_metrics])
    se_depth = sem([metric["Depth"] for metric in nested_metrics])

    mean_leaves = mean([metric["Leaves"] for metric in nested_metrics])
    se_leaves = sem([metric["Leaves"] for metric in nested_metrics])

    print(
        f"  Accuracy: {mean_acc:.4f} ± {se_acc:.4f} | "
        f"Sensitivity: {mean_sens:.4f} ± {se_sens:.4f} | "
        f"Specificity: {mean_spec:.4f} ± {se_spec:.4f} | "
        f"F1: {mean_f1:.4f} ± {se_f1:.4f} | "
        f"AUC: {mean_auc:.4f} ± {se_auc:.4f} | "
        f"Depth: {mean_depth:.2f} ± {se_depth:.2f} | "
        f"Leaves: {mean_leaves:.2f} ± {se_leaves:.2f}"
    )

    all_repeats.append({
        "Repeat": repeat,
        "Criterion": "gini_beta",
        "Mean Accuracy": mean_acc,
        "SE Accuracy": se_acc,
        "Mean Sensitivity": mean_sens,
        "SE Sensitivity": se_sens,
        "Mean Specificity": mean_spec,
        "SE Specificity": se_spec,
        "Mean F1": mean_f1,
        "SE F1": se_f1,
        "Mean AUC": mean_auc,
        "SE AUC": se_auc,
        "Mean Depth": mean_depth,
        "SE Depth": se_depth,
        "Mean Leaves": mean_leaves,
        "SE Leaves": se_leaves
    })


# ----------------------------
# Save Excel & Histograms
# ----------------------------

results_df = pd.DataFrame(all_repeats)

metrics = [
    ("Mean Accuracy", "Accuracy"),
    ("Mean Sensitivity", "Sensitivity"),
    ("Mean Specificity", "Specificity"),
    ("Mean F1", "F1 Score"),
    ("Mean AUC", "AUC"),
    ("Mean Depth", "Depth"),
    ("Mean Leaves", "Leaves")
]

histogram_files = []

for column, label in metrics:
    plt.figure(figsize=(8, 6))
    plt.hist(results_df[column], bins=20, orientation="horizontal")
    plt.title(f"Histogram of {label} Across 1000 Repeats")
    plt.ylabel(label)
    plt.xlabel("Frequency")
    plt.grid(True)
    plt.tight_layout()

    filename = f"Histogram_{label.replace(' ', '_')}.png"
    plt.savefig(filename)
    plt.close()
    histogram_files.append((filename, label))

print("\n✅ Histograms saved as PNG files.")

output_file = "Breast_Cancer_GiniBetaTree_NoFeatureSelection_1000Repeats.xlsx"

with pd.ExcelWriter(output_file, engine="xlsxwriter") as writer:
    results_df.to_excel(writer, sheet_name="Results", index=False)

    workbook = writer.book
    worksheet = workbook.add_worksheet("Histograms")

    row = 0
    column = 0

    for image_file, label in histogram_files:
        worksheet.write(row, column, label)
        worksheet.insert_image(
            row + 1,
            column,
            image_file,
            {"x_scale": 0.7, "y_scale": 0.7}
        )
        row += 20

print(f"\n✅ Excel file saved: '{output_file}'")


# spambase_gini_beta_tree_no_feature_selection.py
# Gini Beta Decision Tree without feature selection

import pandas as pd
import numpy as np
from itertools import product
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_curve, auc
from numpy import mean
from scipy.stats import sem
import matplotlib.pyplot as plt
import xlsxwriter


# ----------------------------
# Utility Functions
# ----------------------------

def preprocess_train_validation(X_train, X_validation):
    """Fit preprocessing on the training fold, then transform both folds."""
    X_train = pd.DataFrame(X_train).copy()
    X_validation = pd.DataFrame(X_validation).copy()

    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [col for col in X_train.columns if col not in numeric_cols]

    if numeric_cols:
        numeric_imputer = SimpleImputer(strategy="mean")
        X_train[numeric_cols] = numeric_imputer.fit_transform(X_train[numeric_cols])
        X_validation[numeric_cols] = numeric_imputer.transform(X_validation[numeric_cols])

    if categorical_cols:
        categorical_imputer = SimpleImputer(strategy="most_frequent")
        X_train[categorical_cols] = categorical_imputer.fit_transform(X_train[categorical_cols])
        X_validation[categorical_cols] = categorical_imputer.transform(X_validation[categorical_cols])

        for column in categorical_cols:
            encoder = LabelEncoder()
            train_values = X_train[column].astype(str)
            encoder.fit(train_values)

            mapping = {label: index for index, label in enumerate(encoder.classes_)}
            X_train[column] = train_values.map(mapping).astype(int)
            X_validation[column] = (
                X_validation[column].astype(str).map(mapping).fillna(-1).astype(int)
            )

    return X_train, X_validation


def evaluate_params(X_train, y_train, X_validation, y_validation,
                    beta, random_state):

    model = DecisionTreeClassifier(
        criterion="gini_beta",
        beta=beta,
        random_state=random_state
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_validation)

    return {
        "beta": beta,
        "score": accuracy_score(y_validation, predictions)
    }


# ----------------------------
# Repeated Experiment
# ----------------------------

grid_values = [0.01, 0.1, 0.2, 0.25, 0.3, 0.35, 0.45, 0.5]
grid_params = [(beta,) for beta in grid_values]

all_repeats = []

for repeat in range(1, 101):
    print(f"\n🔁 Repeat {repeat} of 100")

    data = pd.read_csv("spambase.csv")
    df = pd.DataFrame(data)

    if "spam" not in df.columns:
        raise KeyError("The target column 'spam' was not found in online_shoppers_intention.csv.")

    # Exclude Status from predictors explicitly.
    features = df.drop(columns=["spam"])
    y = df["spam"]

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        features,
        y,
        test_size=0.2,
        stratify=y,
        random_state=repeat
    )

    outer_cv = StratifiedKFold(
        n_splits=10,
        shuffle=True,
        random_state=repeat
    )

    nested_metrics = []

    for fold_number, (train_index, validation_index) in enumerate(
        outer_cv.split(X_train_full, y_train_full), start=1
    ):
        X_train_fold = X_train_full.iloc[train_index]
        X_validation_fold = X_train_full.iloc[validation_index]
        y_train_fold = y_train_full.iloc[train_index]
        y_validation_fold = y_train_full.iloc[validation_index]

        X_train_fold, X_validation_fold = preprocess_train_validation(
            X_train_fold,
            X_validation_fold
        )

        grid_results = Parallel(n_jobs=-1)(
            delayed(evaluate_params)(
                X_train_fold,
                y_train_fold,
                X_validation_fold,
                y_validation_fold,
                beta,
                repeat
            )
            for (beta,) in grid_params
        )

        best_result = max(grid_results, key=lambda result: result["score"])

        final_model = DecisionTreeClassifier(
            criterion="gini_beta",
            beta=best_result["beta"],
            random_state=repeat
        )

        # All columns are used; there is no feature-selection stage.
        final_model.fit(X_train_fold, y_train_fold)

        y_validation_pred = final_model.predict(X_validation_fold)

        alive_index = list(final_model.classes_).index("Alive")
        y_validation_prob = final_model.predict_proba(X_validation_fold)[:, alive_index]

        accuracy = accuracy_score(y_validation_fold, y_validation_pred)
        sensitivity = recall_score(
            y_validation_fold,
            y_validation_pred,
            pos_label=1,
            zero_division=0
        )
        specificity = recall_score(
            y_validation_fold,
            y_validation_pred,
            pos_label= 0,
            zero_division=0
        )
        f1 = f1_score(
            y_validation_fold,
            y_validation_pred,
            pos_label=1,
            zero_division=0
        )
        fpr, tpr, _ = roc_curve(
            y_validation_fold,
            y_validation_prob,
            pos_label=1
        )
        auc_value = auc(fpr, tpr)

        print(
            f"  Fold {fold_number:02d} | "
            f"beta={best_result['beta']} | "
            f"Depth={final_model.get_depth()} | "
            f"Leaves={final_model.get_n_leaves()}"
        )


        nested_metrics.append({
            "accuracy": accuracy,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "f1": f1,
            "auc": auc_value,
            "Depth": final_model.get_depth(),
            "Leaves": final_model.get_n_leaves(),
            "beta": best_result["beta"]
        })

    mean_acc = mean([metric["accuracy"] for metric in nested_metrics])
    se_acc = sem([metric["accuracy"] for metric in nested_metrics])

    mean_sens = mean([metric["sensitivity"] for metric in nested_metrics])
    se_sens = sem([metric["sensitivity"] for metric in nested_metrics])

    mean_spec = mean([metric["specificity"] for metric in nested_metrics])
    se_spec = sem([metric["specificity"] for metric in nested_metrics])

    mean_f1 = mean([metric["f1"] for metric in nested_metrics])
    se_f1 = sem([metric["f1"] for metric in nested_metrics])

    mean_auc = mean([metric["auc"] for metric in nested_metrics])
    se_auc = sem([metric["auc"] for metric in nested_metrics])

    mean_depth = mean([metric["Depth"] for metric in nested_metrics])
    se_depth = sem([metric["Depth"] for metric in nested_metrics])

    mean_leaves = mean([metric["Leaves"] for metric in nested_metrics])
    se_leaves = sem([metric["Leaves"] for metric in nested_metrics])

    print(
        f"  Accuracy: {mean_acc:.4f} ± {se_acc:.4f} | "
        f"Sensitivity: {mean_sens:.4f} ± {se_sens:.4f} | "
        f"Specificity: {mean_spec:.4f} ± {se_spec:.4f} | "
        f"F1: {mean_f1:.4f} ± {se_f1:.4f} | "
        f"AUC: {mean_auc:.4f} ± {se_auc:.4f} | "
        f"Depth: {mean_depth:.2f} ± {se_depth:.2f} | "
        f"Leaves: {mean_leaves:.2f} ± {se_leaves:.2f}"
    )

    all_repeats.append({
        "Repeat": repeat,
        "Criterion": "gini_beta",
        "Mean Accuracy": mean_acc,
        "SE Accuracy": se_acc,
        "Mean Sensitivity": mean_sens,
        "SE Sensitivity": se_sens,
        "Mean Specificity": mean_spec,
        "SE Specificity": se_spec,
        "Mean F1": mean_f1,
        "SE F1": se_f1,
        "Mean AUC": mean_auc,
        "SE AUC": se_auc,
        "Mean Depth": mean_depth,
        "SE Depth": se_depth,
        "Mean Leaves": mean_leaves,
        "SE Leaves": se_leaves
    })


# ----------------------------
# Save Excel & Histograms
# ----------------------------

results_df = pd.DataFrame(all_repeats)

metrics = [
    ("Mean Accuracy", "Accuracy"),
    ("Mean Sensitivity", "Sensitivity"),
    ("Mean Specificity", "Specificity"),
    ("Mean F1", "F1 Score"),
    ("Mean AUC", "AUC"),
    ("Mean Depth", "Depth"),
    ("Mean Leaves", "Leaves")
]

histogram_files = []

for column, label in metrics:
    plt.figure(figsize=(8, 6))
    plt.hist(results_df[column], bins=20, orientation="horizontal")
    plt.title(f"Histogram of {label} Across 100 Repeats")
    plt.ylabel(label)
    plt.xlabel("Frequency")
    plt.grid(True)
    plt.tight_layout()

    filename = f"Histogram_{label.replace(' ', '_')}.png"
    plt.savefig(filename)
    plt.close()
    histogram_files.append((filename, label))

print("\n✅ Histograms saved as PNG files.")

output_file = "spambase_GiniBetaTree_NoFeatureSelection_1000Repeats.xlsx"

with pd.ExcelWriter(output_file, engine="xlsxwriter") as writer:
    results_df.to_excel(writer, sheet_name="Results", index=False)

    workbook = writer.book
    worksheet = workbook.add_worksheet("Histograms")

    row = 0
    column = 0

    for image_file, label in histogram_files:
        worksheet.write(row, column, label)
        worksheet.insert_image(
            row + 1,
            column,
            image_file,
            {"x_scale": 0.7, "y_scale": 0.7}
        )
        row += 20

print(f"\n✅ Excel file saved: '{output_file}'")


# online_shoppers_intention_gini_beta_tree_no_feature_selection.py
# Gini Beta Decision Tree without feature selection

import pandas as pd
import numpy as np
from itertools import product
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_curve, auc
from numpy import mean
from scipy.stats import sem
import matplotlib.pyplot as plt
import xlsxwriter


# ----------------------------
# Utility Functions
# ----------------------------

def preprocess_train_validation(X_train, X_validation):
    """Fit preprocessing on the training fold, then transform both folds."""
    X_train = pd.DataFrame(X_train).copy()
    X_validation = pd.DataFrame(X_validation).copy()

    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [col for col in X_train.columns if col not in numeric_cols]

    if numeric_cols:
        numeric_imputer = SimpleImputer(strategy="mean")
        X_train[numeric_cols] = numeric_imputer.fit_transform(X_train[numeric_cols])
        X_validation[numeric_cols] = numeric_imputer.transform(X_validation[numeric_cols])

    if categorical_cols:
        categorical_imputer = SimpleImputer(strategy="most_frequent")
        X_train[categorical_cols] = categorical_imputer.fit_transform(X_train[categorical_cols])
        X_validation[categorical_cols] = categorical_imputer.transform(X_validation[categorical_cols])

        for column in categorical_cols:
            encoder = LabelEncoder()
            train_values = X_train[column].astype(str)
            encoder.fit(train_values)

            mapping = {label: index for index, label in enumerate(encoder.classes_)}
            X_train[column] = train_values.map(mapping).astype(int)
            X_validation[column] = (
                X_validation[column].astype(str).map(mapping).fillna(-1).astype(int)
            )

    return X_train, X_validation


def evaluate_params(X_train, y_train, X_validation, y_validation,
                    beta, random_state):

    model = DecisionTreeClassifier(
        criterion="gini_beta",
        beta=beta,
        random_state=random_state
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_validation)

    return {
        "beta": beta,
        "score": accuracy_score(y_validation, predictions)
    }


# ----------------------------
# Repeated Experiment
# ----------------------------

grid_values = [0.01, 0.1, 0.2, 0.25, 0.3, 0.35, 0.45, 0.5]
grid_params = [(beta,) for beta in grid_values]

all_repeats = []

for repeat in range(1, 101):
    print(f"\n🔁 Repeat {repeat} of 100")

    data = pd.read_csv("online_shoppers_intention.csv")
    df = pd.DataFrame(data)

    if "Revenue" not in df.columns:
        raise KeyError("The target column 'Revenue' was not found in online_shoppers_intention.csv.")

    # Exclude Status from predictors explicitly.
    features = df.drop(columns=["Revenue"])
    y = df["Revenue"]

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        features,
        y,
        test_size=0.2,
        stratify=y,
        random_state=repeat
    )

    outer_cv = StratifiedKFold(
        n_splits=10,
        shuffle=True,
        random_state=repeat
    )

    nested_metrics = []

    for fold_number, (train_index, validation_index) in enumerate(
        outer_cv.split(X_train_full, y_train_full), start=1
    ):
        X_train_fold = X_train_full.iloc[train_index]
        X_validation_fold = X_train_full.iloc[validation_index]
        y_train_fold = y_train_full.iloc[train_index]
        y_validation_fold = y_train_full.iloc[validation_index]

        X_train_fold, X_validation_fold = preprocess_train_validation(
            X_train_fold,
            X_validation_fold
        )

        grid_results = Parallel(n_jobs=-1)(
            delayed(evaluate_params)(
                X_train_fold,
                y_train_fold,
                X_validation_fold,
                y_validation_fold,
                beta,
                repeat
            )
            for (beta,) in grid_params
        )

        best_result = max(grid_results, key=lambda result: result["score"])

        final_model = DecisionTreeClassifier(
            criterion="gini_beta",
            beta=best_result["beta"],
            random_state=repeat
        )

        # All columns are used; there is no feature-selection stage.
        final_model.fit(X_train_fold, y_train_fold)

        y_validation_pred = final_model.predict(X_validation_fold)

        alive_index = list(final_model.classes_).index("Alive")
        y_validation_prob = final_model.predict_proba(X_validation_fold)[:, alive_index]

        accuracy = accuracy_score(y_validation_fold, y_validation_pred)
        sensitivity = recall_score(
            y_validation_fold,
            y_validation_pred,
            pos_label=True,
            zero_division=0
        )
        specificity = recall_score(
            y_validation_fold,
            y_validation_pred,
            pos_label=False,
            zero_division=0
        )
        f1 = f1_score(
            y_validation_fold,
            y_validation_pred,
            pos_label=True,
            zero_division=0
        )
        fpr, tpr, _ = roc_curve(
            y_validation_fold,
            y_validation_prob,
            pos_label=True
        )
        auc_value = auc(fpr, tpr)

        print(
            f"  Fold {fold_number:02d} | "
            f"beta={best_result['beta']} | "
            f"Depth={final_model.get_depth()} | "
            f"Leaves={final_model.get_n_leaves()}"
        )


        nested_metrics.append({
            "accuracy": accuracy,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "f1": f1,
            "auc": auc_value,
            "Depth": final_model.get_depth(),
            "Leaves": final_model.get_n_leaves(),
            "beta": best_result["beta"]
        })

    mean_acc = mean([metric["accuracy"] for metric in nested_metrics])
    se_acc = sem([metric["accuracy"] for metric in nested_metrics])

    mean_sens = mean([metric["sensitivity"] for metric in nested_metrics])
    se_sens = sem([metric["sensitivity"] for metric in nested_metrics])

    mean_spec = mean([metric["specificity"] for metric in nested_metrics])
    se_spec = sem([metric["specificity"] for metric in nested_metrics])

    mean_f1 = mean([metric["f1"] for metric in nested_metrics])
    se_f1 = sem([metric["f1"] for metric in nested_metrics])

    mean_auc = mean([metric["auc"] for metric in nested_metrics])
    se_auc = sem([metric["auc"] for metric in nested_metrics])

    mean_depth = mean([metric["Depth"] for metric in nested_metrics])
    se_depth = sem([metric["Depth"] for metric in nested_metrics])

    mean_leaves = mean([metric["Leaves"] for metric in nested_metrics])
    se_leaves = sem([metric["Leaves"] for metric in nested_metrics])

    print(
        f"  Accuracy: {mean_acc:.4f} ± {se_acc:.4f} | "
        f"Sensitivity: {mean_sens:.4f} ± {se_sens:.4f} | "
        f"Specificity: {mean_spec:.4f} ± {se_spec:.4f} | "
        f"F1: {mean_f1:.4f} ± {se_f1:.4f} | "
        f"AUC: {mean_auc:.4f} ± {se_auc:.4f} | "
        f"Depth: {mean_depth:.2f} ± {se_depth:.2f} | "
        f"Leaves: {mean_leaves:.2f} ± {se_leaves:.2f}"
    )

    all_repeats.append({
        "Repeat": repeat,
        "Criterion": "gini_beta",
        "Mean Accuracy": mean_acc,
        "SE Accuracy": se_acc,
        "Mean Sensitivity": mean_sens,
        "SE Sensitivity": se_sens,
        "Mean Specificity": mean_spec,
        "SE Specificity": se_spec,
        "Mean F1": mean_f1,
        "SE F1": se_f1,
        "Mean AUC": mean_auc,
        "SE AUC": se_auc,
        "Mean Depth": mean_depth,
        "SE Depth": se_depth,
        "Mean Leaves": mean_leaves,
        "SE Leaves": se_leaves
    })


# ----------------------------
# Save Excel & Histograms
# ----------------------------

results_df = pd.DataFrame(all_repeats)

metrics = [
    ("Mean Accuracy", "Accuracy"),
    ("Mean Sensitivity", "Sensitivity"),
    ("Mean Specificity", "Specificity"),
    ("Mean F1", "F1 Score"),
    ("Mean AUC", "AUC"),
    ("Mean Depth", "Depth"),
    ("Mean Leaves", "Leaves")
]

histogram_files = []

for column, label in metrics:
    plt.figure(figsize=(8, 6))
    plt.hist(results_df[column], bins=20, orientation="horizontal")
    plt.title(f"Histogram of {label} Across 100 Repeats")
    plt.ylabel(label)
    plt.xlabel("Frequency")
    plt.grid(True)
    plt.tight_layout()

    filename = f"Histogram_{label.replace(' ', '_')}.png"
    plt.savefig(filename)
    plt.close()
    histogram_files.append((filename, label))

print("\n✅ Histograms saved as PNG files.")

output_file = "online_shoppers_intention_GiniBetaTree_NoFeatureSelection_100Repeats.xlsx"

with pd.ExcelWriter(output_file, engine="xlsxwriter") as writer:
    results_df.to_excel(writer, sheet_name="Results", index=False)

    workbook = writer.book
    worksheet = workbook.add_worksheet("Histograms")

    row = 0
    column = 0

    for image_file, label in histogram_files:
        worksheet.write(row, column, label)
        worksheet.insert_image(
            row + 1,
            column,
            image_file,
            {"x_scale": 0.7, "y_scale": 0.7}
        )
        row += 20

print(f"\n✅ Excel file saved: '{output_file}'")



# WA_Fn-UseC_-Telco-Customer-Churn_gini_beta_tree_no_feature_selection.py
# Gini Beta Decision Tree without feature selection

import pandas as pd
import numpy as np
from itertools import product
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_curve, auc
from numpy import mean
from scipy.stats import sem
import matplotlib.pyplot as plt
import xlsxwriter


# ----------------------------
# Utility Functions
# ----------------------------

def preprocess_train_validation(X_train, X_validation):
    """Fit preprocessing on the training fold, then transform both folds."""
    X_train = pd.DataFrame(X_train).copy()
    X_validation = pd.DataFrame(X_validation).copy()

    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [col for col in X_train.columns if col not in numeric_cols]

    if numeric_cols:
        numeric_imputer = SimpleImputer(strategy="mean")
        X_train[numeric_cols] = numeric_imputer.fit_transform(X_train[numeric_cols])
        X_validation[numeric_cols] = numeric_imputer.transform(X_validation[numeric_cols])

    if categorical_cols:
        categorical_imputer = SimpleImputer(strategy="most_frequent")
        X_train[categorical_cols] = categorical_imputer.fit_transform(X_train[categorical_cols])
        X_validation[categorical_cols] = categorical_imputer.transform(X_validation[categorical_cols])

        for column in categorical_cols:
            encoder = LabelEncoder()
            train_values = X_train[column].astype(str)
            encoder.fit(train_values)

            mapping = {label: index for index, label in enumerate(encoder.classes_)}
            X_train[column] = train_values.map(mapping).astype(int)
            X_validation[column] = (
                X_validation[column].astype(str).map(mapping).fillna(-1).astype(int)
            )

    return X_train, X_validation


def evaluate_params(X_train, y_train, X_validation, y_validation,
                    beta, random_state):

    model = DecisionTreeClassifier(
        criterion="gini_beta",
        beta=beta,
        random_state=random_state
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_validation)

    return {
        "beta": beta,
        "score": accuracy_score(y_validation, predictions)
    }


# ----------------------------
# Repeated Experiment
# ----------------------------

grid_values = [0.01, 0.1, 0.2, 0.25, 0.3, 0.35, 0.45, 0.5]
grid_params = [(beta,) for beta in grid_values]

all_repeats = []

for repeat in range(1, 101):
    print(f"\n🔁 Repeat {repeat} of 100")

    data = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df = pd.DataFrame(data)

    if "Churn" not in df.columns:
        raise KeyError(
            "The target column 'Churn' was not found in "
            "WA_Fn-UseC_-Telco-Customer-Churn.csv."
        )

    # Exclude the target column from predictors explicitly.
    features = df.drop(
        columns=["Churn", "customerID"],
        errors="ignore"
    )
    y = df["Churn"].astype(str).str.strip()

    valid_labels = {"Yes", "No"}
    unexpected_labels = set(y.unique()) - valid_labels
    if unexpected_labels:
        raise ValueError(
            f"Unexpected values in Churn: {sorted(unexpected_labels)}. "
            "Expected only Yes and No."
        )

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        features,
        y,
        test_size=0.2,
        stratify=y,
        random_state=repeat
    )

    outer_cv = StratifiedKFold(
        n_splits=10,
        shuffle=True,
        random_state=repeat
    )

    nested_metrics = []

    for fold_number, (train_index, validation_index) in enumerate(
        outer_cv.split(X_train_full, y_train_full), start=1
    ):
        X_train_fold = X_train_full.iloc[train_index]
        X_validation_fold = X_train_full.iloc[validation_index]
        y_train_fold = y_train_full.iloc[train_index]
        y_validation_fold = y_train_full.iloc[validation_index]

        X_train_fold, X_validation_fold = preprocess_train_validation(
            X_train_fold,
            X_validation_fold
        )

        grid_results = Parallel(n_jobs=-1)(
            delayed(evaluate_params)(
                X_train_fold,
                y_train_fold,
                X_validation_fold,
                y_validation_fold,
                beta,
                repeat
            )
            for (beta,) in grid_params
        )

        best_result = max(grid_results, key=lambda result: result["score"])

        final_model = DecisionTreeClassifier(
            criterion="gini_beta",
            beta=best_result["beta"],
            random_state=repeat
        )

        # All columns are used; there is no feature-selection stage.
        final_model.fit(X_train_fold, y_train_fold)

        y_validation_pred = final_model.predict(X_validation_fold)

        positive_index = list(final_model.classes_).index("Yes")
        y_validation_prob = final_model.predict_proba(X_validation_fold)[:, positive_index]

        accuracy = accuracy_score(y_validation_fold, y_validation_pred)
        sensitivity = recall_score(
            y_validation_fold,
            y_validation_pred,
            pos_label="Yes",
            zero_division=0
        )
        tn, fp, fn, tp = confusion_matrix(
            y_validation_fold,
            y_validation_pred,
            labels=["No", "Yes"]
        ).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = f1_score(
            y_validation_fold,
            y_validation_pred,
            pos_label="Yes",
            zero_division=0
        )
        fpr, tpr, _ = roc_curve(
            y_validation_fold,
            y_validation_prob,
            pos_label="Yes"
        )
        auc_value = auc(fpr, tpr)

        print(
            f"  Fold {fold_number:02d} | "
            f"beta={best_result['beta']} | "
            f"Depth={final_model.get_depth()} | "
            f"Leaves={final_model.get_n_leaves()}"
        )


        nested_metrics.append({
            "accuracy": accuracy,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "f1": f1,
            "auc": auc_value,
            "Depth": final_model.get_depth(),
            "Leaves": final_model.get_n_leaves(),
            "beta": best_result["beta"]
        })

    mean_acc = mean([metric["accuracy"] for metric in nested_metrics])
    se_acc = sem([metric["accuracy"] for metric in nested_metrics])

    mean_sens = mean([metric["sensitivity"] for metric in nested_metrics])
    se_sens = sem([metric["sensitivity"] for metric in nested_metrics])

    mean_spec = mean([metric["specificity"] for metric in nested_metrics])
    se_spec = sem([metric["specificity"] for metric in nested_metrics])

    mean_f1 = mean([metric["f1"] for metric in nested_metrics])
    se_f1 = sem([metric["f1"] for metric in nested_metrics])

    mean_auc = mean([metric["auc"] for metric in nested_metrics])
    se_auc = sem([metric["auc"] for metric in nested_metrics])

    mean_depth = mean([metric["Depth"] for metric in nested_metrics])
    se_depth = sem([metric["Depth"] for metric in nested_metrics])

    mean_leaves = mean([metric["Leaves"] for metric in nested_metrics])
    se_leaves = sem([metric["Leaves"] for metric in nested_metrics])

    print(
        f"  Accuracy: {mean_acc:.4f} ± {se_acc:.4f} | "
        f"Sensitivity: {mean_sens:.4f} ± {se_sens:.4f} | "
        f"Specificity: {mean_spec:.4f} ± {se_spec:.4f} | "
        f"F1: {mean_f1:.4f} ± {se_f1:.4f} | "
        f"AUC: {mean_auc:.4f} ± {se_auc:.4f} | "
        f"Depth: {mean_depth:.2f} ± {se_depth:.2f} | "
        f"Leaves: {mean_leaves:.2f} ± {se_leaves:.2f}"
    )

    all_repeats.append({
        "Repeat": repeat,
        "Criterion": "gini_beta",
        "Mean Accuracy": mean_acc,
        "SE Accuracy": se_acc,
        "Mean Sensitivity": mean_sens,
        "SE Sensitivity": se_sens,
        "Mean Specificity": mean_spec,
        "SE Specificity": se_spec,
        "Mean F1": mean_f1,
        "SE F1": se_f1,
        "Mean AUC": mean_auc,
        "SE AUC": se_auc,
        "Mean Depth": mean_depth,
        "SE Depth": se_depth,
        "Mean Leaves": mean_leaves,
        "SE Leaves": se_leaves
    })


# ----------------------------
# Save Excel & Histograms
# ----------------------------

results_df = pd.DataFrame(all_repeats)

metrics = [
    ("Mean Accuracy", "Accuracy"),
    ("Mean Sensitivity", "Sensitivity"),
    ("Mean Specificity", "Specificity"),
    ("Mean F1", "F1 Score"),
    ("Mean AUC", "AUC"),
    ("Mean Depth", "Depth"),
    ("Mean Leaves", "Leaves")
]

histogram_files = []

for column, label in metrics:
    plt.figure(figsize=(8, 6))
    plt.hist(results_df[column], bins=20, orientation="horizontal")
    plt.title(f"Histogram of {label} Across 100 Repeats")
    plt.ylabel(label)
    plt.xlabel("Frequency")
    plt.grid(True)
    plt.tight_layout()

    filename = f"Histogram_{label.replace(' ', '_')}.png"
    plt.savefig(filename)
    plt.close()
    histogram_files.append((filename, label))

print("\n✅ Histograms saved as PNG files.")

output_file = "WA_Fn-UseC_-Telco-Customer-Churn_GiniBetaTree_NoFeatureSelection_100Repeats.xlsx"

with pd.ExcelWriter(output_file, engine="xlsxwriter") as writer:
    results_df.to_excel(writer, sheet_name="Results", index=False)

    workbook = writer.book
    worksheet = workbook.add_worksheet("Histograms")

    row = 0
    column = 0

    for image_file, label in histogram_files:
        worksheet.write(row, column, label)
        worksheet.insert_image(
            row + 1,
            column,
            image_file,
            {"x_scale": 0.7, "y_scale": 0.7}
        )
        row += 20

print(f"\n✅ Excel file saved: '{output_file}'")


# TUANDROMD_gini_beta_tree_no_feature_selection.py
# Gini Beta Decision Tree without feature selection

import pandas as pd
import numpy as np
from itertools import product
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_curve, auc
from numpy import mean
from scipy.stats import sem
import matplotlib.pyplot as plt
import xlsxwriter


# ----------------------------
# Utility Functions
# ----------------------------

def preprocess_train_validation(X_train, X_validation):
    """Fit preprocessing on the training fold, then transform both folds."""
    X_train = pd.DataFrame(X_train).copy()
    X_validation = pd.DataFrame(X_validation).copy()

    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [col for col in X_train.columns if col not in numeric_cols]

    if numeric_cols:
        numeric_imputer = SimpleImputer(strategy="mean")
        X_train[numeric_cols] = numeric_imputer.fit_transform(X_train[numeric_cols])
        X_validation[numeric_cols] = numeric_imputer.transform(X_validation[numeric_cols])

    if categorical_cols:
        categorical_imputer = SimpleImputer(strategy="most_frequent")
        X_train[categorical_cols] = categorical_imputer.fit_transform(X_train[categorical_cols])
        X_validation[categorical_cols] = categorical_imputer.transform(X_validation[categorical_cols])

        for column in categorical_cols:
            encoder = LabelEncoder()
            train_values = X_train[column].astype(str)
            encoder.fit(train_values)

            mapping = {label: index for index, label in enumerate(encoder.classes_)}
            X_train[column] = train_values.map(mapping).astype(int)
            X_validation[column] = (
                X_validation[column].astype(str).map(mapping).fillna(-1).astype(int)
            )

    return X_train, X_validation


def evaluate_params(X_train, y_train, X_validation, y_validation,
                    beta, random_state):

    model = DecisionTreeClassifier(
        criterion="gini_beta",
        beta=beta,
        random_state=random_state
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_validation)

    return {
        "beta": beta,
        "score": accuracy_score(y_validation, predictions)
    }


# ----------------------------
# Repeated Experiment
# ----------------------------

grid_values = [0.01, 0.1, 0.2, 0.25, 0.3, 0.35, 0.45, 0.5]
grid_params = [(beta,) for beta in grid_values]

all_repeats = []

for repeat in range(1, 101):
    print(f"\n🔁 Repeat {repeat} of 100")

    data = pd.read_csv("TUANDROMD.csv")
    df = pd.DataFrame(data)

    if "Label" not in df.columns:
        raise KeyError("The target column 'Label' was not found in online_shoppers_intention.csv.")

    # Exclude Status from predictors explicitly.
    features = df.drop(columns=["Label"])
    y = df["Label"]

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        features,
        y,
        test_size=0.2,
        stratify=y,
        random_state=repeat
    )

    outer_cv = StratifiedKFold(
        n_splits=10,
        shuffle=True,
        random_state=repeat
    )

    nested_metrics = []

    for fold_number, (train_index, validation_index) in enumerate(
        outer_cv.split(X_train_full, y_train_full), start=1
    ):
        X_train_fold = X_train_full.iloc[train_index]
        X_validation_fold = X_train_full.iloc[validation_index]
        y_train_fold = y_train_full.iloc[train_index]
        y_validation_fold = y_train_full.iloc[validation_index]

        X_train_fold, X_validation_fold = preprocess_train_validation(
            X_train_fold,
            X_validation_fold
        )

        grid_results = Parallel(n_jobs=-1)(
            delayed(evaluate_params)(
                X_train_fold,
                y_train_fold,
                X_validation_fold,
                y_validation_fold,
                beta,
                repeat
            )
            for (beta,) in grid_params
        )

        best_result = max(grid_results, key=lambda result: result["score"])

        final_model = DecisionTreeClassifier(
            criterion="gini_beta",
            beta=best_result["beta"],
            random_state=repeat
        )

        # All columns are used; there is no feature-selection stage.
        final_model.fit(X_train_fold, y_train_fold)

        y_validation_pred = final_model.predict(X_validation_fold)

        positive_index = list(final_model.classes_).index(1)
        y_validation_prob = final_model.predict_proba(X_validation_fold)[:, positive_index]

        accuracy = accuracy_score(y_validation_fold, y_validation_pred)
        sensitivity = recall_score(
            y_validation_fold,
            y_validation_pred,
            pos_label=1,
            zero_division=0
        )
        tn, fp, fn, tp = confusion_matrix(
            y_validation_fold,
            y_validation_pred,
            labels=[0, 1]
        ).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = f1_score(
            y_validation_fold,
            y_validation_pred,
            pos_label=1,
            zero_division=0
        )
        fpr, tpr, _ = roc_curve(
            y_validation_fold,
            y_validation_prob,
            pos_label=1
        )
        auc_value = auc(fpr, tpr)

        print(
            f"  Fold {fold_number:02d} | "
            f"beta={best_result['beta']} | "
            f"Depth={final_model.get_depth()} | "
            f"Leaves={final_model.get_n_leaves()}"
        )


        nested_metrics.append({
            "accuracy": accuracy,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "f1": f1,
            "auc": auc_value,
            "Depth": final_model.get_depth(),
            "Leaves": final_model.get_n_leaves(),
            "beta": best_result["beta"]
        })

    mean_acc = mean([metric["accuracy"] for metric in nested_metrics])
    se_acc = sem([metric["accuracy"] for metric in nested_metrics])

    mean_sens = mean([metric["sensitivity"] for metric in nested_metrics])
    se_sens = sem([metric["sensitivity"] for metric in nested_metrics])

    mean_spec = mean([metric["specificity"] for metric in nested_metrics])
    se_spec = sem([metric["specificity"] for metric in nested_metrics])

    mean_f1 = mean([metric["f1"] for metric in nested_metrics])
    se_f1 = sem([metric["f1"] for metric in nested_metrics])

    mean_auc = mean([metric["auc"] for metric in nested_metrics])
    se_auc = sem([metric["auc"] for metric in nested_metrics])

    mean_depth = mean([metric["Depth"] for metric in nested_metrics])
    se_depth = sem([metric["Depth"] for metric in nested_metrics])

    mean_leaves = mean([metric["Leaves"] for metric in nested_metrics])
    se_leaves = sem([metric["Leaves"] for metric in nested_metrics])

    print(
        f"  Accuracy: {mean_acc:.4f} ± {se_acc:.4f} | "
        f"Sensitivity: {mean_sens:.4f} ± {se_sens:.4f} | "
        f"Specificity: {mean_spec:.4f} ± {se_spec:.4f} | "
        f"F1: {mean_f1:.4f} ± {se_f1:.4f} | "
        f"AUC: {mean_auc:.4f} ± {se_auc:.4f} | "
        f"Depth: {mean_depth:.2f} ± {se_depth:.2f} | "
        f"Leaves: {mean_leaves:.2f} ± {se_leaves:.2f}"
    )

    all_repeats.append({
        "Repeat": repeat,
        "Mean Accuracy": mean_acc,
        "SE Accuracy": se_acc,
        "Mean Sensitivity": mean_sens,
        "SE Sensitivity": se_sens,
        "Mean Specificity": mean_spec,
        "SE Specificity": se_spec,
        "Mean F1": mean_f1,
        "SE F1": se_f1,
        "Mean AUC": mean_auc,
        "SE AUC": se_auc,
        "Mean Depth": mean_depth,
        "SE Depth": se_depth,
        "Mean Leaves": mean_leaves,
        "SE Leaves": se_leaves
    })


# ----------------------------
# Save Excel & Histograms
# ----------------------------

results_df = pd.DataFrame(all_repeats)

metrics = [
    ("Mean Accuracy", "Accuracy"),
    ("Mean Sensitivity", "Sensitivity"),
    ("Mean Specificity", "Specificity"),
    ("Mean F1", "F1 Score"),
    ("Mean AUC", "AUC"),
    ("Mean Depth", "Depth"),
    ("Mean Leaves", "Leaves")
]

histogram_files = []

for column, label in metrics:
    plt.figure(figsize=(8, 6))
    plt.hist(results_df[column], bins=20, orientation="horizontal")
    plt.title(f"Histogram of {label} Across 100 Repeats")
    plt.ylabel(label)
    plt.xlabel("Frequency")
    plt.grid(True)
    plt.tight_layout()

    filename = f"Histogram_{label.replace(' ', '_')}.png"
    plt.savefig(filename)
    plt.close()
    histogram_files.append((filename, label))

print("\n✅ Histograms saved as PNG files.")

output_file = "TUANDROMD_GiniBetaTree_NoFeatureSelection_100Repeats.xlsx"

with pd.ExcelWriter(output_file, engine="xlsxwriter") as writer:
    results_df.to_excel(writer, sheet_name="Results", index=False)

    workbook = writer.book
    worksheet = workbook.add_worksheet("Histograms")

    row = 0
    column = 0

    for image_file, label in histogram_files:
        worksheet.write(row, column, label)
        worksheet.insert_image(
            row + 1,
            column,
            image_file,
            {"x_scale": 0.7, "y_scale": 0.7}
        )
        row += 20

print(f"\n✅ Excel file saved: '{output_file}'")


