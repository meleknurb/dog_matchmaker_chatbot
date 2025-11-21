# data_loader.py
import pandas as pd

def load_breed_data():
    return pd.read_csv('data/breed_traits.csv')

def load_trait_descriptions():
    return pd.read_csv('data/trait_description.csv')