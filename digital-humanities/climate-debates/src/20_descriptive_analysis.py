import os
import pandas as pd
import matplotlib.pyplot as plt
from config import REDUCED_PARLAMINT_DIR

OUTPUT_DIR = f"{os.path.abspath(os.path.dirname(__file__))}/../output/20/"

os.makedirs(OUTPUT_DIR, exist_ok=True)

all_speeches_df = pd.read_csv(
    f"{REDUCED_PARLAMINT_DIR}/all_speeches.csv", low_memory=False
)

print("speeches loaded")

# Topic

topic_counts = all_speeches_df["Topic"].value_counts()
topic_counts.to_csv(f"{OUTPUT_DIR}/topic_distribution.csv")

topic_counts.plot.barh(logx=True)
plt.savefig(f"{OUTPUT_DIR}/topic_distribution.png", bbox_inches="tight", dpi=300)
plt.close()

print("topic distribution table and plot saved")

# Party_orientation

party_counts = all_speeches_df["Party_orientation"].value_counts()
party_counts.to_csv(f"{OUTPUT_DIR}/party_orientation.csv")

party_counts.plot.barh(logx=True)
plt.savefig(f"{OUTPUT_DIR}/party_orientation.png", bbox_inches="tight", dpi=300)
plt.close()

print("party orientation table and plot saved")

# Party_status

party_counts = all_speeches_df["Party_status"].value_counts()
party_counts.to_csv(f"{OUTPUT_DIR}/party_status.csv")

party_counts.plot.barh(logx=True)
plt.savefig(f"{OUTPUT_DIR}/party_status.png", bbox_inches="tight", dpi=300)
plt.close()

print("party status table and plot saved")

# Speaker_gender

gender_counts = all_speeches_df["Speaker_gender"].value_counts()
gender_counts.to_csv(f"{OUTPUT_DIR}/speaker_gender.csv")

gender_counts.plot.barh(logx=True)
plt.savefig(f"{OUTPUT_DIR}/speaker_gender.png", bbox_inches="tight", dpi=300)
plt.close()

print("speaker gender table and plot saved")
