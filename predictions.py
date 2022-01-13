train_df = pd.read_csv("train_df.csv")
test_df = pd.read_csv("test_df.csv")

test_x = test_df.values
y = train_df["change_type"].values
train_df = train_df.drop("change_type", axis=1)

