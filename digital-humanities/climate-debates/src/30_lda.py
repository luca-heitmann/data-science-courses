import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
import matplotlib.pyplot as plt
from config import REDUCED_PARLAMINT_DIR, PARLAMINT_DIR, OCED_EPS_FILE, SEED
from pyconll.conllu import conllu
from tqdm import tqdm

OUTPUT_DIR = f"{os.path.abspath(os.path.dirname(__file__))}/../output/30/"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Corpus vorbereiten
_text_cache = {}


def get_speech(country: str, year: int, textId: str, docId: str) -> str:
    global _text_cache

    cache_key = f"{country}_{year}_{textId}_{docId}"
    if cache_key in _text_cache:
        return _text_cache[cache_key]

    filepath = f"{PARLAMINT_DIR}/ParlaMint-{country}-en.ana/ParlaMint-{country}-en.conllu/{year}/{textId}.conllu"

    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    content = []
    found = False

    for line in lines:
        # Prüfe ob neue Document-ID beginnt
        if line.strip().startswith("# newdoc id ="):
            # Extrahiere die ID aus der Zeile
            current_id = line.split("=", 1)[1].strip()

            if current_id == docId:
                # Ziel-ID gefunden, beginne Sammlung
                found = True
                content.append(line)
            elif found:
                # Eine neue ID gefunden, aber wir haben schon gesammelt -> Ende
                break
        elif found:
            # Sammle alle Zeilen nach der Ziel-ID
            content.append(line)

    doc_text = "".join(content)

    session_doc = conllu.load_from_string(doc_text)

    doc = ""

    for sentence in session_doc:
        doc += " ".join(
            [str(token.lemma) for token in sentence.tokens if token.upos in ["NOUN"]]
        )  # , 'PROPN'
        doc += "\n"

    _text_cache[cache_key] = doc

    return doc


def get_documents_for_speeches(speeches: pd.DataFrame):
    documents = []
    for _, row in tqdm(speeches.iterrows(), total=len(speeches)):
        text = get_speech(row["country"], row["year"], row["Text_ID"], row["ID"])
        if text:  # Nur nicht-leere Texte
            documents.append(text)
    return documents


# 2. LDA-Modell auf Gesamtcorpus trainieren
def train_lda_model(documents, n_topics=10, max_features=5000):
    # 1. Standard-Liste holen
    stop_words = list(ENGLISH_STOP_WORDS)

    # 2. Eigene Wörter hinzufügen
    custom_words = []
    stop_words.extend(custom_words)

    # # Vektorisierung
    # vectorizer = CountVectorizer(
    #     max_features=max_features,
    #     min_df=15,  # Wörter müssen in mind. 5 Dokumenten vorkommen
    #     max_df=0.7,  # Wörter dürfen in max. 70% der Dokumente vorkommen
    #     stop_words="english",  # Oder 'german' falls deutsch
    # )
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=0.01,
        max_df=0.5,
        stop_words=stop_words,
        ngram_range=(1, 2),
    )

    # Die Matrix enthält nun TF-IDF Werte statt reiner Counts
    doc_term_matrix = vectorizer.fit_transform(documents)

    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=SEED,
        max_iter=10,
        n_jobs=-1,
        learning_method="online",  # Oft stabiler bei TF-IDF Matrizen
        doc_topic_prior=0.005,
    )

    lda.fit(doc_term_matrix)

    return lda, vectorizer


# 3. Topics interpretieren
def get_top_words(model, vectorizer, n_top_words=10):
    feature_names = vectorizer.get_feature_names_out()
    topic_desc = ""

    for topic_idx, topic in enumerate(model.components_):
        top_indices = topic.argsort()[-n_top_words:][::-1]
        top_words = [feature_names[i] for i in top_indices]
        topic_desc += f"Topic {topic_idx}: {', '.join(top_words)}\n"

    return topic_desc


# 4. Topic-Verteilung für Subcorpus berechnen
def get_topic_distribution(documents, lda, vectorizer):
    doc_term_matrix = vectorizer.transform(documents)
    topic_distributions = lda.transform(doc_term_matrix)

    # Durchschnittliche Topic-Verteilung über alle Dokumente
    avg_distribution = topic_distributions.mean(axis=0)

    return avg_distribution, topic_distributions


# 5. Subcorpora vergleichen
def compare_subcorpora(speeches_df, lda, vectorizer, group_column):
    """
    Vergleicht Topic-Verteilungen zwischen verschiedenen Gruppen

    speeches_df: DataFrame mit allen Reden
    lda: trainiertes LDA-Modell
    vectorizer: trainierter Vectorizer
    group_column: Spalte zum Gruppieren (z.B. "Party_orientation")
    """
    results = {}

    for group_name in speeches_df[group_column].unique():
        # Subcorpus erstellen
        subcorpus_df = speeches_df[speeches_df[group_column] == group_name]
        documents = get_documents_for_speeches(subcorpus_df)

        # Topic-Verteilung berechnen
        avg_dist, _ = get_topic_distribution(documents, lda, vectorizer)
        results[group_name] = avg_dist

    return results


# 6. Visualisierung
def plot_topic_comparison(results, save_path=""):
    """
    Visualisiert Topic-Verteilungen für verschiedene Gruppen
    """
    groups = list(results.keys())
    n_topics = len(results[groups[0]])

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(n_topics)
    width = 0.8 / len(groups)

    for i, group in enumerate(groups):
        offset = width * i - (width * len(groups) / 2)
        ax.bar(x + offset, results[group], width, label=group)

    ax.set_xlabel("Topic")
    ax.set_ylabel("Durchschnittliche Wahrscheinlichkeit")
    ax.set_title("Topic-Verteilung nach Gruppen")
    ax.set_xticks(x)
    ax.set_xticklabels([f"T{i}" for i in range(n_topics)])
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")


###################################################


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

# Filter auf Reden zum Thema Umwelt
environment_speeches = merged_df[merged_df["Topic"] == "Environment"]
all_documents = get_documents_for_speeches(environment_speeches)

# LDA-Modell trainieren
lda_model, vectorizer = train_lda_model(all_documents, n_topics=10)

# Topics anschauen
print("Gefundene Topics:")
top_words = get_top_words(lda_model, vectorizer, n_top_words=20)
print(top_words)

with open(f"{OUTPUT_DIR}/topics.txt", "w") as f:
    f.writelines(top_words)

# Vergleich nach EPS
topic_comparison = compare_subcorpora(
    environment_speeches, lda_model, vectorizer, group_column="EPS_category"
)
plot_topic_comparison(
    topic_comparison, save_path=f"{OUTPUT_DIR}/topic_comparison_eps.png"
)

# Vergleich nach Party_orientation
topic_comparison = compare_subcorpora(
    environment_speeches, lda_model, vectorizer, group_column="Party_orientation"
)
plot_topic_comparison(
    topic_comparison, save_path=f"{OUTPUT_DIR}/topic_comparison_party_orientation.png"
)

# Vergleich nach Party_status
topic_comparison = compare_subcorpora(
    environment_speeches, lda_model, vectorizer, group_column="Party_status"
)
plot_topic_comparison(
    topic_comparison, save_path=f"{OUTPUT_DIR}/topic_comparison_party_status.png"
)

# Vergleich nach Party_orientation_filtered
topic_comparison = compare_subcorpora(
    environment_speeches[
        environment_speeches["Party_orientation"].isin(["Left", "Centre", "Right"])
    ],
    lda_model,
    vectorizer,
    group_column="Party_orientation",
)
plot_topic_comparison(
    topic_comparison,
    save_path=f"{OUTPUT_DIR}/topic_comparison_party_orientation_filtered.png",
)

# Vergleich nach country
topic_comparison = compare_subcorpora(
    environment_speeches,
    lda_model,
    vectorizer,
    group_column="country",
)
plot_topic_comparison(
    topic_comparison, save_path=f"{OUTPUT_DIR}/topic_comparison_country.png"
)

# Vergleich nach year
topic_comparison = compare_subcorpora(
    environment_speeches,
    lda_model,
    vectorizer,
    group_column="year",
)
plot_topic_comparison(
    topic_comparison, save_path=f"{OUTPUT_DIR}/topic_comparison_year.png"
)

# Vergleich nach gender
topic_comparison = compare_subcorpora(
    environment_speeches,
    lda_model,
    vectorizer,
    group_column="Speaker_gender",
)
plot_topic_comparison(
    topic_comparison, save_path=f"{OUTPUT_DIR}/topic_comparison_speaker_gender.png"
)

# Vergleich nach alter
env_cleaned = environment_speeches.copy()
env_cleaned = env_cleaned[env_cleaned["Speaker_birth"] != "-"]
env_cleaned["Speaker_birth"] = env_cleaned["Speaker_birth"].astype(int)
min_y = env_cleaned["Speaker_birth"].min()
max_y = env_cleaned["Speaker_birth"].max()
bins = np.arange(min_y, max_y + 16, 15)
env_cleaned["Birth_Bin"] = pd.cut(env_cleaned["Speaker_birth"], bins=bins, right=False)

topic_comparison = compare_subcorpora(
    env_cleaned,
    lda_model,
    vectorizer,
    group_column="Birth_Bin",
)
plot_topic_comparison(
    topic_comparison, save_path=f"{OUTPUT_DIR}/topic_comparison_birth_Bin.png"
)
