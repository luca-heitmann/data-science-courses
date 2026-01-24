import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from config import PARLAMINT_DIR, OCED_EPS_FILE, REDUCED_PARLAMINT_DIR

OUTPUT_DIR = f"{os.path.abspath(os.path.dirname(__file__))}/../output/50/"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_speech_sentiment(country: str, year: int, textId: str, docId: str):
    filepath = f"{PARLAMINT_DIR}/ParlaMint-{country}-en.ana/ParlaMint-{country}-en.conllu/{year}/{textId}-ana-meta.tsv"
    df = pd.read_csv(filepath, sep="\t")
    df = df[df["Parent_ID"] == docId]
    senti_counts = df["Senti_3"].value_counts(normalize=True)
    positiv = senti_counts.get("Positive", 0)
    neutral = senti_counts.get("Neutral", 0)
    negativ = senti_counts.get("Negative", 0)
    return (positiv, neutral, negativ)


def get_sentiments(speeches: pd.DataFrame):
    speeches = speeches.copy()
    speeches["positive"] = 0.0
    speeches["neutral"] = 0.0
    speeches["negative"] = 0.0

    for idx, row in tqdm(speeches.iterrows(), total=len(speeches)):
        senti_counts = get_speech_sentiment(
            row["country"], row["year"], row["Text_ID"], row["ID"]
        )
        speeches.loc[idx, ["positive", "neutral", "negative"]] = senti_counts  # type: ignore

    return speeches


# ============================================================================
# VERWENDUNG
# ============================================================================

all_speeches_df = pd.read_csv(
    f"{REDUCED_PARLAMINT_DIR}/all_speeches.csv", low_memory=False
)
eps_df = pd.read_csv(OCED_EPS_FILE)

country_mapper = {
    "AT": "Austria",
    "BE": "Belgium",
    "BG": "Bulgaria",
    "HR": "Croatia",
    "CZ": "Czechia",
    "DK": "Denmark",
    "EE": "Estonia",
    "FI": "Finland",
    "FR": "France",
    "DE": "Germany",
    "GR": "Greece",
    "HU": "Hungary",
    "IS": "Iceland",
    "IT": "Italy",
    "LV": "Latvia",
    "NL": "Netherlands",
    "NO": "Norway",
    "PL": "Poland",
    "PT": "Portugal",
    "RS": "Serbia",
    "SI": "Slovenia",
    "ES": "Spain",
    "SE": "Sweden",
    "TR": "Türkiye",
    "GB": "United Kingdom",
    "UA": "Ukraine",
    "BA": "Bosnia and Herzegovina",
    # Spanische Regionen -> Spanien
    "ES-PV": "Spain",
    "ES-CT": "Spain",
    "ES-GA": "Spain",
}

all_speeches_filtered = all_speeches_df[
    (all_speeches_df["year"] >= 2017) & (all_speeches_df["year"] <= 2020)
].copy()

eps_filtered = eps_df[
    (eps_df["TIME_PERIOD"] >= 2017) & (eps_df["TIME_PERIOD"] <= 2020)
].copy()

# Mappe ParlaMint Ländercodes zu vollen Namen
all_speeches_filtered["country_full"] = all_speeches_filtered["country"].map(
    country_mapper
)

# Berechne EPS Score pro Land und Jahr (Summe über alle Variablen)
eps_yearly = (
    eps_filtered.groupby(["Country", "TIME_PERIOD"])["OBS_VALUE"].sum().reset_index()
)
eps_yearly.columns = ["Country", "year", "EPS_Score"]

# Merge beide DataFrames auf Land UND Jahr
merged_df = all_speeches_filtered.merge(
    eps_yearly,
    left_on=["country_full", "year"],
    right_on=["Country", "year"],
    how="left",
)
merged_df = merged_df[merged_df["Country"].notna()]
merged_df["EPS_category"] = pd.qcut(
    merged_df["EPS_Score"], q=3, labels=["low", "mid", "high"]
)

environment_speeches = merged_df[merged_df["Topic"] == "Environment"]
environment_speeches = get_sentiments(environment_speeches)


environment_speeches[["positive", "neutral", "negative"]].mean().plot.barh()
plt.savefig(f"{OUTPUT_DIR}/sentiments_all.png", dpi=300, bbox_inches="tight")

env_cleaned = environment_speeches.copy()
env_cleaned = env_cleaned[env_cleaned["Speaker_birth"] != "-"]
env_cleaned["Speaker_birth"] = env_cleaned["Speaker_birth"].astype(int)
min_y = env_cleaned["Speaker_birth"].min()
max_y = env_cleaned["Speaker_birth"].max()
bins = np.arange(min_y, max_y + 16, 15)
env_cleaned["Birth_Bin"] = pd.cut(env_cleaned["Speaker_birth"], bins=bins, right=False)

env_cleaned.groupby(by="Birth_Bin", observed=True).agg(
    {
        "positive": "mean",
        "neutral": "mean",
        "negative": "mean",
    }
).sort_index().plot.barh()
plt.savefig(f"{OUTPUT_DIR}/sentiments_age.png", dpi=300, bbox_inches="tight")

environment_speeches.groupby(by="EPS_category", observed=True).agg(
    {
        "positive": "mean",
        "neutral": "mean",
        "negative": "mean",
    }
).sort_index().plot.barh()
plt.savefig(f"{OUTPUT_DIR}/sentiments_eps.png", dpi=300, bbox_inches="tight")


environment_speeches[
    environment_speeches["Party_orientation"].isin(["Left", "Centre", "Right"])
].groupby(by="Party_orientation", observed=True).agg(
    {
        "positive": "mean",
        "neutral": "mean",
        "negative": "mean",
    }
).sort_index().plot.barh()
plt.savefig(
    f"{OUTPUT_DIR}/sentiments_party_orientation.png", dpi=300, bbox_inches="tight"
)


environment_speeches.groupby(by="Party_status", observed=True).agg(
    {
        "positive": "mean",
        "neutral": "mean",
        "negative": "mean",
    }
).sort_index().plot.barh()
plt.savefig(f"{OUTPUT_DIR}/sentiments_party_status.png", dpi=300, bbox_inches="tight")


environment_speeches.groupby(by="country", observed=True).agg(
    {
        "positive": "mean",
        "neutral": "mean",
        "negative": "mean",
    }
).sort_index().plot.barh()
plt.savefig(f"{OUTPUT_DIR}/sentiments_country.png", dpi=300, bbox_inches="tight")
