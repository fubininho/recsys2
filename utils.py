import sys
import pandas as pd

def read_jsonl(data):
   return pd.read_json(data,lines=True)

def read_csv(data):
   return pd.read_csv(data)