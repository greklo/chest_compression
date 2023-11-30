import pandas as pd
import numpy as np

from collections import OrderedDict
from itertools import product

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    auc,
    brier_score_loss,
)
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

from scipy import stats

import matplotlib.pyplot as plt


color_cols = ["#d9327f", "#4fabc9", "#2a428a"]

def model_stats(
    y_train, train_prob, y_test, test_prob, test_pred, json_path=None, label = 'is_compression',
    ) -> pd.DataFrame:
    """
    Calculate model metrics using the true labels, predictions, and probabilities.

    Parameters:
    -----------------
    `y_train`: Labels for training dataset.
    `train_prob`: Predicted label estimate between 0 and 1 for the observations in the training dataset.
    `y_test`: Labels for the test dataset.
    `test_prob`: Predicted label estimate between 0 and 1 for the observations in the test dataset.
    `test_pred`: Predicted label estimate (0 or 1) for the observations in the test dataset.

    Returns:
    -------------
    A pandas dataframe containg the accuracy, precision, recall, F1, ROC AUC, 1- FPR, 1 - FNR,
    brier gain, 1- KS, and lift statistic.
    """

    # Traditional metrics
    d = OrderedDict()
    d["accuracy"] = accuracy_score(y_test, test_pred)
    d["precision"] = precision_score(y_test, test_pred)
    d["recall"] = recall_score(y_test, test_pred)
    d["f1"] = f1_score(y_test, test_pred)
    d["roc_auc"] = roc_auc_score(y_test, test_prob)

    # Normalized confusion matrix
    cm = confusion_matrix(y_test, test_pred)
    norm_cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    tnr, fpr, fnr, tpr = norm_cm.ravel()
    d["tpr"] = tpr
    d["tnr"] = tnr
    d["1-fpr"] = 1 - fpr
    d["1-fnr"] = 1 - fnr

    # Brier gain and KS score
    d["brier_gain"] = 1 - brier_score_loss(y_test, test_prob)
    ks = stats.ks_2samp(test_prob[y_test == 1], train_prob[y_train == 1])
    d["1-ks"] = 1 - ks[0]

    # Create df with test labels and probabilities and then create lift table from them
    df_y = pd.DataFrame(y_test, columns=[label])
    df_y["probability"] = test_prob
    lift_df = lift_table(df_y, label, "probability")

    # Convert gain back to float, use np.trapz to find gain AUC, and then normalize
    gain = (
        lift_df["Cumulative Percent of All Positive Timelines"]
        .str.rstrip("%")
        .astype("float")
    )
    gain_auc = np.trapz(gain) / (100 * len(gain))
    d["lift_statistic"] = gain_auc / (1 - (np.sum(y_test == 1) / y_test.shape[0]))

    # Return df
    if json_path is not None:
        pd.DataFrame([d]).to_json(path_or_buf=json_path)
    return pd.DataFrame([d])

def lift_table(df, target, score, number_bins=10, duplicates="raise") -> pd.DataFrame:
    """
    Creates a lift table for the evaluation of a predictive model.

    Parameters:
    -----------------
    `df`: 
    `target`: The name of the variable to predict.
    `score`: 
    `number_bins`: Number of bins to use for the lift chart. The default is deciles (10). 
    `duplicates`: Not use.

    Returns:
    -----------------
    A dataframe with decile, minimum and maximum probability, total timelines, positive class proportion,
    percent of all positive timelines, cumulative percent of all positive timelines, and lift.
    """

    # Group the data into n equal sized groups
    # The grouping is done by the predicted probability
    df["negative"] = 1 - df[target]
    df.sort_values(score, ascending=False, inplace=True)
    df["idx"] = range(1, len(df) + 1)
    df["bins"] = pd.cut(
        df["idx"], bins=number_bins, right=True, retbins=False, precision=3
    )

    # Obtain summary information for each group
    aggregated = df.groupby("bins", as_index=False)
    lift_table = pd.DataFrame(np.vstack(aggregated.min()[score]), columns=["min_score"])
    lift_table.sort_values("min_score", ascending=False, inplace=True)
    lift_table["Decile"] = np.arange(1, (len(df.bins.unique()) + 1))

    # Add probabilities and timeline count
    lift_table["Minimum Probability"] = (100 * aggregated.min()[score]).map(
        "{:,.0f}%".format
    )
    lift_table["Maximum Probability"] = (100 * aggregated.max()[score]).map(
        "{:,.0f}%".format
    )
    timelines = aggregated.sum()[target] + aggregated.sum()["negative"]
    lift_table["Total Timelines"] = timelines.map("{:,}".format)

    # Calculate positive class proportions and percent of positive timelines
    lift_table["Positive Class Proportion"] = (
        100 * aggregated.sum()[target] / timelines
    ).map("{:,.0f}%".format)
    pct_positive_all_timelines = (
        aggregated.sum()[target] / aggregated.sum()[target].sum()
    )
    lift_table["Percent Of All Positive Timelines"] = (
        100 * pct_positive_all_timelines
    ).map("{:,.0f}%".format)

    # Calculate cumulative positve class proportion (gain) and lift
    cum_pct_positive = 100 * pct_positive_all_timelines.cumsum()
    lift_table["Cumulative Percent of All Positive Timelines"] = (cum_pct_positive).map(
        "{:,.0f}%".format
    )
    lift_table["Lift"] = cum_pct_positive / (lift_table["Decile"] * (100 / number_bins))

    return lift_table.drop("min_score", axis=1)


def make_ks_plot(
    y_train: pd.Series,
    train_proba: np.ndarray,
    y_test: pd.Series,
    test_proba: np.ndarray,
    bins=30,
    fig_sz=(10, 8),
    label = "label",
):
    """
    Creates  KS test/train overtraining plots for classifier output.
    
    Parameters:
    ------------
    `y_train`: Predicted probabilities from the training dataset.
    `train_proba`: np.ndarray from sklearn predict_praba(). Same shape as y_train. 0-1 probabilities from model.
    `y_test`: Series with outputs of model from the test dataset.
    `test_proba`: np.ndarray from sklearn predict_praba(). Same shape as y_test. 0-1 probabilities from model.
    `bins`: number of bins for visualization. Default is 30.
    `label_col_name`: name of y-label. Change to whatever your model has it named. Default 'label'.
    `fig_sz`: change to True in order to get larger outputs. Default False.
    
    Returns:
    ------------
    The KS test/train overtraining plots for classifier output.
    """

    train = pd.DataFrame(y_train, columns=[label])
    test = pd.DataFrame(y_test, columns=[label])
    train["probability"] = train_proba
    test["probability"] = test_proba

    decisions = []
    for df in [train, test]:
        d1 = df["probability"][df[label] == 1]
        d2 = df["probability"][df[label] == 0]
        decisions += [d1, d2]

    low = min(np.min(d) for d in decisions)
    high = max(np.max(d) for d in decisions)
    low_high = (low, high)

    with plt.style.context(("seaborn-whitegrid")):
        fig = plt.figure(figsize=fig_sz)

        train_pos = plt.hist(
            decisions[0],
            color=color_cols[1],
            alpha=0.5,
            range=low_high,
            bins=bins,
            histtype="stepfilled",
            density=True,
            label="+ (train)",
        )

        train_neg = plt.hist(
            decisions[1],
            color=color_cols[0],
            alpha=0.5,
            range=low_high,
            bins=bins,
            histtype="stepfilled",
            density=True,
            label="- (train)",
        )

        hist, bins = np.histogram(decisions[2], bins=bins, range=low_high, density=True)
        scale = len(decisions[2]) / sum(hist)
        err = np.sqrt(hist * scale) / scale

        width = bins[1] - bins[0]
        center = (bins[:-1] + bins[1:]) / 2
        test_pos = plt.errorbar(
            center, hist, yerr=err, fmt="o", c=color_cols[1], label="+ (test)"
        )

        hist, bins = np.histogram(decisions[3], bins=bins, range=low_high, density=True)
        scale = len(decisions[2]) / sum(hist)
        err = np.sqrt(hist * scale) / scale

        test_neg = plt.errorbar(
            center, hist, yerr=err, fmt="o", c=color_cols[0], label="- (test)"
        )

        # get the KS score
        ks = stats.ks_2samp(decisions[0], decisions[2])

        plt.xlabel("Classifier Output", fontsize=12)
        plt.ylabel("Arbitrary Normalized Units", fontsize=12)

        plt.xlim(0, 1)
        plt.plot(
            [],
            [],
            " ",
            label="KS Statistic (p-value) :"
            + str(round(ks[0], 2))
            + "("
            + str(round(ks[1], 2))
            + ")",
        )
        plt.legend(loc="best", fontsize=12)
        plt.show()
        plt.close()