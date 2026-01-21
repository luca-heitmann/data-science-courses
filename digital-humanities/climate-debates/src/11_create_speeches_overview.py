import os
import re
import glob
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from config import REDUCED_PARLAMINT_DIR

OUTPUT_DIR = f"{os.path.abspath(os.path.dirname(__file__))}/../output/11/"

os.makedirs(OUTPUT_DIR, exist_ok=True)

all_speeches_df = pd.read_csv(
    f"{REDUCED_PARLAMINT_DIR}/all_speeches.csv", low_memory=False
)

print("speeches loaded")
