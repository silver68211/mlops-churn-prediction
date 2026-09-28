from src.make_data import generate_data

data = generate_data(
    n_samples=3000,
    random_state=100
)

data.to_csv(
    "data/retraining/churn_latest.csv",
    index=False
)

print(data.shape)