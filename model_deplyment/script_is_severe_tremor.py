
import argparse
import joblib
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, precision_score, f1_score

def model_fn(model_dir):
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf

if __name__ == "__main__":

    print("extracting arguments")
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    # to simplify the model I don't use all sklearn LogisticRegression hyperparameters
    parser.add_argument("--penalty", type=str, default='l1')
    parser.add_argument("--solver", type=str, default='liblinear')
    parser.add_argument("--C", type=int, default=0.75)
    parser.add_argument("--max_iter", type=int, default=50)

    # Data, model, and output directories
    parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR'))
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST"))
    parser.add_argument("--train_file", type=str, default="X_train_is_severe_tremor.csv")
    parser.add_argument("--test_file", type=str, default="X_test_is_severe_tremor.csv")
    parser.add_argument(
        "--features", type=str
    )  # in this script we ask user to explicitly name features
    parser.add_argument(
        "--target", type=str
    )  # in this script we ask user to explicitly name the target
    

    args, _ = parser.parse_known_args()

    print("reading data")
    train_df = pd.read_csv(os.path.join(args.train, args.train_file))
    test_df = pd.read_csv(os.path.join(args.test, args.test_file))

    print("building training and testing datasets")
    X_train = train_df[args.features.split()]
    X_test = test_df[args.features.split()]
    y_train = train_df[args.target]
    y_test = test_df[args.target]
    

    # train
    print("training model")
    model = LogisticRegression(
        random_state=7, solver=args.solver, penalty=args.penalty, C=args.C, max_iter=args.max_iter
    )
    
    model.fit(X_train, y_train)

    # print performance metrics
    print("validating model")
    y_pred = model.predict(X_test)
    F_score = f1_score(y_test, y_pred).round(2)
    Precision = precision_score(y_test, y_pred, average = None).round(2)
    Recall = recall_score(y_test, y_pred, average = None).round(2)  
    print("F1=" + str(F_score))
    print("Recall=" + str(Recall[1]))
    print("Precision=" + str(Precision[1]))

    # persist model
    path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(model, path)
    print("model persisted at " + path)
