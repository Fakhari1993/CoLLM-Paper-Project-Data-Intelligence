
import numpy as np
import pandas as pd

save_path = "/content/"

train_ = pd.read_pickle(save_path + "train_ood2.pkl")
valid_ = pd.read_pickle(save_path + "valid_ood2.pkl")
test_  = pd.read_pickle(save_path + "test_ood2.pkl")

print("Shapes:")
print("train_:", train_.shape)
print("valid_:", valid_.shape)
print("test_ :", test_.shape)

user_info = train_.groupby('uid').agg({"label": "count"})
print("\nUser interaction stats (before filtering):")
print(user_info.describe())

train_user = np.array(list(user_info.index))

user_threshold = user_info["label"].quantile(0.2)
print("\nChosen user_threshold (20% quantile):", user_threshold)

user_info = user_info[user_info["label"] > user_threshold]
print("\nUser interaction stats (after filtering > 20% quantile):")
print(user_info.describe())
print("Number of warm users:", user_info.shape[0])

warm_user = np.array(list(user_info.index))

item_info = train_.groupby('iid').agg({"label": "count"})
print("\nItem interaction stats (before filtering):")
print(item_info.describe())

train_item = np.array(list(item_info.index))

item_threshold = item_info["label"].quantile(0.2)
print("\nChosen item_threshold (20% quantile):", item_threshold)

item_info = item_info[item_info["label"] > item_threshold]
print("\nItem interaction stats (after filtering > 20% quantile):")
print(item_info.describe())
print("Number of warm items:", item_info.shape[0])

warm_item = np.array(list(item_info.index))

print("\nShapes:")
print("warm_user :", warm_user.shape)
print("warm_item :", warm_item.shape)
print("train_user:", train_user.shape)
print("train_item:", train_item.shape)

test_["warm"] = test_[["uid", "iid"]].apply(
    lambda x: (x.uid in warm_user) and (x.iid in warm_item),
    axis=1
).astype("int")

test_["cold"] = test_[["uid", "iid"]].apply(
    lambda x: (x.uid not in train_user) and (x.iid not in train_item),
    axis=1
).astype("int")

test_["not_cold"] = (1 - test_["cold"]).astype("int")

print("\nNew columns added to test_: ['warm', 'cold', 'not_cold']")
print("test_ shape:", test_.shape)

print("\nHead of test_:")
print(test_.head())

print("\nDistribution of not_cold / warm / cold:")
print(test_[["not_cold", "warm", "cold"]].describe())

out_path = save_path + "test_warm_cold_ood2.pkl"
test_.to_pickle(out_path)
print("\nSaved:", out_path)
