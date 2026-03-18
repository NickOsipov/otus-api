"""
Train pipeline for the project
"""

import os
from typing import Optional, Tuple

import pandas as pd
from loguru import logger
from sklearn.base import BaseEstimator
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from src.inference import save_model


def train_model(
    train: pd.DataFrame,
    model: BaseEstimator = RandomForestClassifier,
    model_params: dict = {"n_estimators": 100},
) -> RandomForestClassifier:
    """
    Function to train a model on the training data.

    Parameters
    ----------
    model : BaseEstimator, optional
        The model to train, by default RandomForestClassifier.
    train : pd.DataFrame
        The training data to train the model on.

    Returns
    -------
    RandomForestClassifier
        The trained model.
    """
    X = train.drop("target", axis=1)
    y = train["target"]
    clf = model(**model_params)
    clf.fit(X, y)
    return clf


def evaluate_model(model: RandomForestClassifier, test: pd.DataFrame) -> float:
    """
    Function to evaluate the model on the test data.

    Parameters
    ----------
    model : RandomForestClassifier
        The trained model to evaluate.
    test : pd.DataFrame
        The test data to evaluate the model on.

    Returns
    -------
    float
        The accuracy of the model on the test data.
    """
    X = test.drop("target", axis=1)
    y = test["target"]
    y_pred = model.predict(X)
    return accuracy_score(y, y_pred)


def load_data() -> pd.DataFrame:
    """
    Function to load the iris dataset from sklearn and return it as a pandas DataFrame.

    Returns
    -------
    pd.DataFrame
        The iris dataset as a pandas DataFrame.

    Notes
    -----
    The iris dataset is a classic dataset in machine learning and is used to demonstrate.
    """
    iris = load_iris()
    # pylint: disable=no-member
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df = df.rename(
        columns={
            "sepal length (cm)": "sepal_length",
            "sepal width (cm)": "sepal_width",
            "petal length (cm)": "petal_length",
            "petal width (cm)": "petal_width",
        }
    )
    df["target"] = iris.target
    return df


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: Optional[pd.Series] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Function to split the data into training and test sets.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the data to split.
    test_size : float, optional
        The proportion of the dataset to include in the test split, by default 0.2.
    random_state : int, optional
        The random state to use, by default 42.
    stratify : Optional, optional
        The variable to stratify the data on, by default None.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        A tuple containing the training and test DataFrames.
    """
    train, test = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=stratify
    )
    return train, test


def main() -> None:
    """
    Train pipeline for the project
    """
    logger.info("-----------------")
    logger.info("Starting pipeline")
    logger.info("Loading data")
    df = load_data()

    logger.info("Splitting data")
    train, test = split_data(df)

    logger.info("Training model")
    model = train_model(train)

    logger.info("Saving model")
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "model.joblib")
    save_model(model, model_path)

    logger.info("Evaluating model")
    accuracy = evaluate_model(model, test)
    logger.info(f"Model accuracy: {accuracy}")

    logger.info("Pipeline complete")


if __name__ == "__main__":
    main()
