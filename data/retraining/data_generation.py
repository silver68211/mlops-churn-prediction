from pathlib import Path

from src.make_data import generate_data


ROOT = Path(__file__).resolve().parents[1]

OUTPUT_PATH = (
    ROOT
    / "data"
    / "retraining"
    / "churn_latest.csv"
)


def main():

    data = generate_data(
        n_samples=3000,
        random_state=100
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    data.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(
        f"Saved retraining data to: {OUTPUT_PATH}"
    )

    print(
        f"Shape: {data.shape}"
    )


if __name__ == "__main__":
    main()