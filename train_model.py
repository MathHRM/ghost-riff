import pickle
import argparse

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--hand", choices=["chord", "stroke", "both"], required=True)
args = parser.parse_args()


def train(hand):
    data_dict = pickle.load(open(f"dataset_{hand}.pickle", "rb"))

    data = np.asarray(data_dict["data"])
    labels = np.asarray(data_dict["labels"])

    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

    model = RandomForestClassifier()
    model.fit(x_train, y_train)

    y_predict = model.predict(x_test)
    score = accuracy_score(y_predict, y_test)
    print(f"{hand}: {score * 100:.1f}% of samples classified correctly")

    output = f"model_{hand}.pickle"
    with open(output, "wb") as f:
        pickle.dump({"model": model}, f)

    print(f"{output} saved")


hands = ["chord", "stroke"] if args.hand == "both" else [args.hand]
for hand in hands:
    train(hand)
