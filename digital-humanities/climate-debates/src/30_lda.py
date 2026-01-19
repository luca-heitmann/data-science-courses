import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
import matplotlib.pyplot as plt
from config import REDUCED_PARLAMINT_DIR

OUTPUT_DIR = f"{os.path.abspath(os.path.dirname(__file__))}/../output/30/"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Corpus vorbereiten
_text_cache = {}


def get_speech(session_path: str, speech_id: str) -> str:
    global _text_cache
    filepath = f"{REDUCED_PARLAMINT_DIR}{session_path}"
    if filepath not in _text_cache:
        _text_cache[filepath] = {}
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split(None, 1)
                    if len(parts) == 2:
                        _text_cache[filepath][parts[0]] = parts[1]

    return _text_cache[filepath].get(speech_id)


def get_documents_for_speeches(speeches: pd.DataFrame):
    documents = []
    for _, row in speeches.iterrows():
        text = get_speech(row["speech_path"], row["ID"])
        if text:  # Nur nicht-leere Texte
            documents.append(text)
    return documents


# 2. LDA-Modell auf Gesamtcorpus trainieren
def train_lda_model(documents, n_topics=10, max_features=5000):
    # Vektorisierung
    vectorizer = CountVectorizer(
        max_features=max_features,
        min_df=5,  # Wörter müssen in mind. 5 Dokumenten vorkommen
        max_df=0.7,  # Wörter dürfen in max. 70% der Dokumente vorkommen
        stop_words="english",  # Oder 'german' falls deutsch
    )

    doc_term_matrix = vectorizer.fit_transform(documents)

    # LDA trainieren
    lda = LatentDirichletAllocation(
        n_components=n_topics, random_state=42, max_iter=20, n_jobs=-1
    )

    lda.fit(doc_term_matrix)

    return lda, vectorizer


# 3. Topics interpretieren
def print_top_words(model, vectorizer, n_top_words=10):
    feature_names = vectorizer.get_feature_names_out()
    topic_desc = ""

    for topic_idx, topic in enumerate(model.components_):
        top_indices = topic.argsort()[-n_top_words:][::-1]
        top_words = [feature_names[i] for i in top_indices]
        topic_desc += f"Topic {topic_idx}: {', '.join(top_words)}\n"

    print(topic_desc)
    with open(f"{OUTPUT_DIR}topics.txt", "w") as f:
        f.writelines(topic_desc)


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
def plot_topic_comparison(results, save_path):
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


# VERWENDUNG:
# ============

# 1. Gesamtcorpus vorbereiten
all_speeches_df = pd.read_csv(
    f"{REDUCED_PARLAMINT_DIR}/all_speeches.csv", low_memory=False
)
environment_speeches = all_speeches_df[all_speeches_df["Topic"] == "Environment"]
all_documents = get_documents_for_speeches(environment_speeches)

# 2. LDA-Modell trainieren
lda_model, vectorizer = train_lda_model(all_documents, n_topics=10)

# 3. Topics anschauen
print("Gefundene Topics:")
print_top_words(lda_model, vectorizer, n_top_words=10)

# 4. Subcorpora vergleichen (z.B. nach Parteiausrichtung)
party_comparison = compare_subcorpora(
    environment_speeches, lda_model, vectorizer, group_column="Party_orientation"
)

# 5. Ergebnisse visualisieren
plot_topic_comparison(
    party_comparison, save_path=f"{OUTPUT_DIR}/topic_comparison_party.png"
)

# 6. Auch andere Gruppierungen möglich:
# Nach Regierungsparty
party_comparison = compare_subcorpora(
    environment_speeches,
    lda_model,
    vectorizer,
    group_column="Party_status",
)
plot_topic_comparison(
    party_comparison, save_path=f"{OUTPUT_DIR}/topic_comparison_party_status.png"
)

# Nach Party filtered
party_comparison = compare_subcorpora(
    environment_speeches[
        environment_speeches["Party_orientation"].isin(["Left", "Centre", "Right"])
    ],
    lda_model,
    vectorizer,
    group_column="Party_orientation",
)
plot_topic_comparison(
    party_comparison, save_path=f"{OUTPUT_DIR}/topic_comparison_party_filtered.png"
)

# Nach Land
country_comparison = compare_subcorpora(
    environment_speeches, lda_model, vectorizer, group_column="country"
)
plot_topic_comparison(
    country_comparison, save_path=f"{OUTPUT_DIR}/topic_comparison_country.png"
)

# Nach Jahr
year_comparison = compare_subcorpora(
    environment_speeches, lda_model, vectorizer, group_column="year"
)
plot_topic_comparison(
    year_comparison, save_path=f"{OUTPUT_DIR}/topic_comparison_year.png"
)

# Nach Geschlecht
gender_comparison = compare_subcorpora(
    environment_speeches, lda_model, vectorizer, group_column="Speaker_gender"
)
plot_topic_comparison(
    gender_comparison, save_path=f"{OUTPUT_DIR}/topic_comparison_gender.png"
)

# 7. Statistischer Vergleich: Welches Topic ist am wichtigsten für jede Gruppe?
for group, distribution in party_comparison.items():
    dominant_topic = np.argmax(distribution)
    print(
        f"{group}: Dominantes Topic = {dominant_topic} ({distribution[dominant_topic]:.3f})"
    )
