import pandas as pd


def prepare_data() -> tuple:
    df = pd.read_csv("data/train.csv")


if __name__ == "__main__":
    prepare_data()