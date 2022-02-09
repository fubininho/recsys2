import sys
import pandas as pd
from utils import read_jsonl, read_csv

# Loading the data
ratings = read_jsonl(sys.argv[1])
print(ratings.head(5))
print("")

content = read_jsonl(sys.argv[2])
print(content.head(5))
print("")

targets = read_csv(sys.argv[3])
print(targets.head(5))
print("")

# Exploring the data
print(content.columns)
print(content["BoxOffice"].unique())
print("")
# print(content.dropna(['Episode', 'Season'], axis=1))

# Cleaning content
# content = content.drop(['Website', 'DVD', 'Response'], axis=1)

# Funk SVD prediction
from surprise import Dataset, Reader, SVD
reader = Reader(rating_scale=(0, 10))
data = ratings.drop(columns=['Timestamp'])
data = Dataset.load_from_df(data,reader)

algo = SVD()
algo.fit(data.build_full_trainset())
targets["Prediction"] = 0
predictions = []
for row in targets.to_dict("records"):
   predictions.append(algo.predict(uid=row["UserId"],iid=row["ItemId"]).est)

targets["Prediction"] = predictions
print(targets.head(5))

targets = targets.sort_values(["UserId", "Prediction"], ascending = (True, False))
targets = targets.set_index("UserId")
targets = targets.drop(columns=['Prediction'])

targets.to_csv("submission.csv")

# Plot predction

# Genre predction

# Mixed Predction

# Predict Targets

# Rank Targests by user, Take predctions out for the result



 

