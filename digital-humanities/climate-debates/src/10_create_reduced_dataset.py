import os
import re
import glob
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from config import REDUCED_PARLAMINT_DIR, PARLAMINT_DIR

OUTPUT_DIR = f"{os.path.abspath(os.path.dirname(__file__))}/../output/10/"

selected_years = range(2017, 2023)  # 2017-2022

os.makedirs(OUTPUT_DIR, exist_ok=True)

countries = [
    match.group(1)
    for f in glob.glob(f"{PARLAMINT_DIR}/ParlaMint-*-en.ana")
    if (match := re.match(r"^ParlaMint-(.*)-en\.ana$", os.path.basename(f)))
]

print(f"{len(countries)} countries found")

years_in_country = {}

for c in countries:
    path = f"{PARLAMINT_DIR}/ParlaMint-{c}-en.ana/ParlaMint-{c}-en.txt/*"
    years = [
        match.group(1)
        for f in glob.glob(path)
        if (match := re.match(r"^(\d\d\d\d)$", os.path.basename(f)))
    ]
    years_in_country[c] = years

all_years = sorted(set(year for years in years_in_country.values() for year in years))
print(f"years found: {all_years[0]} - {all_years[-1]}")

matrix = np.zeros((len(countries), len(all_years)))

for i, country in enumerate(countries):
    for j, year in enumerate(all_years):
        if year in years_in_country[country]:
            matrix[i, j] = 1

plt.figure(figsize=(16, 6))
sns.heatmap(
    matrix,
    xticklabels=all_years,
    yticklabels=countries,
    cmap="YlOrRd",
    cbar_kws={"label": "Daten vorhanden"},
    linewidths=0.5,
    linecolor="gray",
)

plt.xlabel("Jahr", fontsize=12)
plt.ylabel("Land", fontsize=12)
plt.title("Verfügbarkeit von Daten pro Land und Jahr", fontsize=14, pad=20)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/available_data.png")

print("created available data plot")

all_speeches = []

for c in countries:
    for y in selected_years:
        path = f"{PARLAMINT_DIR}/ParlaMint-{c}-en.ana/ParlaMint-{c}-en.txt/{y}"
        speeches = glob.glob(f"{path}/*.txt")
        for s in speeches:
            match = re.match(r"^(.*)\.txt$", os.path.basename(s))

            if not match:
                print(f"Unable to find text ID for {s}")
                continue

            all_speeches.append(
                {
                    "text_id": match.group(1),
                    "country": c,
                    "year": y,
                    "speech_path": s.replace(PARLAMINT_DIR, ""),
                    "meta_path": f"{path}/{match.group(1)}-meta.tsv".replace(
                        PARLAMINT_DIR, ""
                    ),
                    "ana_meta_path": f"{path}/{match.group(1)}-ana-meta.tsv".replace(
                        PARLAMINT_DIR, ""
                    ),
                }
            )

print(f"{len(all_speeches)} protocols found")

for speech in all_speeches:
    for key in ["speech_path", "meta_path", "ana_meta_path"]:
        source = PARLAMINT_DIR + speech[key]
        destination = REDUCED_PARLAMINT_DIR + speech[key]
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        shutil.copy(source, destination)

print("protocols copied")

first_session = True
speeches_table_path = f"{REDUCED_PARLAMINT_DIR}/all_speeches.csv"

if os.path.exists(speeches_table_path):
    os.remove(speeches_table_path)

for i, session in enumerate(all_speeches):
    speeches_df = pd.read_csv(f"{PARLAMINT_DIR}/{session['meta_path']}", sep="\t")
    speeches_df["country"] = session["country"]
    speeches_df["year"] = session["year"]
    speeches_df["speech_path"] = session["speech_path"]
    speeches_df.to_csv(speeches_table_path, header=first_session, index=False, mode="a")
    first_session = False

print("created speeches table")
