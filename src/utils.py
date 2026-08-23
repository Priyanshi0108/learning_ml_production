import pickle
import os


def load_object(file_path):
    with open(file_path,"rb") as file:
        return pickle.load(file)