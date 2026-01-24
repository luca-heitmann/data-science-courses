import os
from config import REDUCED_PARLAMINT_DIR, PARLAMINT_DIR, OCED_EPS_FILE, SEED
from pyconll.conllu import conllu
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import networkx as nx
import matplotlib.pyplot as plt
import re
from scipy.sparse import csr_matrix


OUTPUT_DIR = f"{os.path.abspath(os.path.dirname(__file__))}/../output/40/"

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
            [str(token.lemma) for token in sentence.tokens]
        )  # if token.upos in ['NOUN']]) #, 'PROPN'
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


# ============================================================================
# 1. SATZ-SEGMENTIERUNG
# ============================================================================


def split_into_sentences(documents):
    """
    Teilt Dokumente in Sätze auf (analog zu corpus_reshape in R)
    """
    sentences = []
    for doc in documents:
        # Einfache Satz-Segmentierung basierend auf Satzzeichen
        sents = re.split(r"[.!?]+", doc)
        sents = [s.strip() for s in sents if s.strip()]
        sentences.extend(sents)
    return sentences


# ============================================================================
# 2. BINÄRE DOCUMENT-TERM-MATRIX ERSTELLEN
# ============================================================================


def create_binary_dtm(sentences, min_df=10, stop_words="english"):
    """
    Erstellt eine binäre Document-Term-Matrix

    Parameters:
    -----------
    sentences : list
        Liste von Sätzen
    min_df : int
        Minimale Dokumentfrequenz (wie minimumFrequency in R)
    stop_words : str oder list
        Stop Words ('english' für sklearn's Liste)

    Returns:
    --------
    bin_dtm : scipy.sparse matrix
        Binäre Document-Term-Matrix
    feature_names : list
        Liste der Terme
    """
    # CountVectorizer mit binärer Ausgabe
    vectorizer = CountVectorizer(
        min_df=min_df,
        stop_words=stop_words,
        binary=True,  # Binäre Werte: 1 wenn Term vorkommt, 0 sonst
        lowercase=True,
        token_pattern=r"\b[a-z]+\b",  # Nur Buchstaben
    )

    bin_dtm = vectorizer.fit_transform(sentences)
    feature_names = vectorizer.get_feature_names_out()

    print(f"Anzahl Sätze: {len(sentences)}")
    print(f"Anzahl Features: {len(feature_names)}")

    return bin_dtm, feature_names


# ============================================================================
# 3. CO-OCCURRENCE MATRIX BERECHNEN
# ============================================================================


def compute_cooccurrence_matrix(bin_dtm):
    """
    Berechnet Co-Occurrence Matrix durch Matrixmultiplikation
    (analog zu: coocCounts <- t(binDTM) %*% binDTM)

    Returns:
    --------
    cooc_matrix : numpy array
        Term-Term Co-Occurrence Matrix
    """
    # Transponierte Matrix multipliziert mit Original
    cooc_matrix = (bin_dtm.T @ bin_dtm).toarray()
    return cooc_matrix


# ============================================================================
# 4. SIGNIFIKANZ-MASSE BERECHNEN
# ============================================================================


def calculate_cooc_statistics(
    cooc_term, bin_dtm, feature_names, cooc_matrix, measure="loglik"
):
    """
    Berechnet Co-Occurrence Signifikanz für einen Zielterm

    Parameters:
    -----------
    cooc_term : str
        Zielterm
    bin_dtm : scipy.sparse matrix
        Binäre DTM
    feature_names : list
        Feature-Namen
    cooc_matrix : numpy array
        Co-Occurrence Matrix
    measure : str
        'loglik', 'dice', oder 'mi'

    Returns:
    --------
    dict : {term: significance_score}
    """
    if cooc_term not in feature_names:
        raise ValueError(f"Term '{cooc_term}' nicht im Vokabular")

    # Index des Zielterms
    term_idx = list(feature_names).index(cooc_term)

    # Notwendige Zählungen
    k = bin_dtm.shape[0]  # Anzahl Dokumente (Sätze)
    ki = bin_dtm[:, term_idx].sum()  # Häufigkeit des Zielterms
    kj = bin_dtm.sum(axis=0).A1  # Häufigkeiten aller Terme
    kij = cooc_matrix[term_idx, :]  # Co-Occurrences mit Zielterm

    # Verhindere Division durch Null
    kj = np.where(kj == 0, 1, kj)
    kij = np.where(kij == 0, 1, kij)

    if measure.lower() == "loglik":
        # Log-Likelihood
        sig = 2 * (
            (k * np.log(k))
            - (ki * np.log(ki))
            - (kj * np.log(kj))
            + (kij * np.log(kij))
            + (k - ki - kj + kij) * np.log(np.maximum(k - ki - kj + kij, 1))
            + (ki - kij) * np.log(np.maximum(ki - kij, 1))
            + (kj - kij) * np.log(np.maximum(kj - kij, 1))
            - (k - ki) * np.log(k - ki)
            - (k - kj) * np.log(k - kj)
        )
    elif measure.lower() == "dice":
        # Dice Coefficient
        sig = 2 * kij / (ki + kj)
    elif measure.lower() == "mi":
        # Mutual Information
        sig = np.log(k * kij / (ki * kj))
    else:
        raise ValueError("measure muss 'loglik', 'dice' oder 'mi' sein")

    # Als Dictionary zurückgeben, sortiert nach Signifikanz
    result = dict(zip(feature_names, sig))
    result = {k: v for k, v in sorted(result.items(), key=lambda x: x[1], reverse=True)}

    return result


# ============================================================================
# 5. NETZWERK-GRAPH ERSTELLEN
# ============================================================================


def build_cooccurrence_network(
    cooc_term, bin_dtm, feature_names, cooc_matrix, num_coocs=15, measure="loglik"
):
    """
    Erstellt Netzwerk-Graph für Co-Occurrences
    (analog zum R-Code mit resultGraph)

    Returns:
    --------
    pd.DataFrame mit Spalten: from, to, sig
    """
    result_graph = []

    # 1. Co-Occurrences für Zielterm berechnen
    coocs = calculate_cooc_statistics(
        cooc_term, bin_dtm, feature_names, cooc_matrix, measure
    )

    # Top N Co-Occurrences
    top_coocs = dict(list(coocs.items())[:num_coocs])

    # Kanten vom Zielterm zu seinen Co-Occurrences
    for term, sig in top_coocs.items():
        result_graph.append({"from": cooc_term, "to": term, "sig": sig})

    # 2. Sekundäre Co-Occurrences (für jeden Co-Occurrence Term)
    for i, (new_cooc_term, _) in enumerate(top_coocs.items()):
        if i >= num_coocs:
            break

        coocs2 = calculate_cooc_statistics(
            new_cooc_term, bin_dtm, feature_names, cooc_matrix, measure
        )

        top_coocs2 = dict(list(coocs2.items())[:num_coocs])

        # Überspringe erste (ist der Term selbst)
        for j, (term, sig) in enumerate(top_coocs2.items()):
            if j == 0:  # Überspringe Selbst-Referenz
                continue
            result_graph.append({"from": new_cooc_term, "to": term, "sig": sig})

    return pd.DataFrame(result_graph)


# ============================================================================
# 6. NETZWERK VISUALISIEREN
# ============================================================================


def visualize_cooccurrence_network(
    result_graph, cooc_term, min_degree=2, figsize=(16, 12), save_path=""
):
    """
    Visualisiert das Co-Occurrence Netzwerk
    """
    # NetworkX Graph erstellen
    G = nx.from_pandas_edgelist(
        result_graph,
        source="from",
        target="to",
        edge_attr="sig",
        create_using=nx.Graph(),  # type: ignore
    )

    # Knoten mit zu wenig Verbindungen entfernen
    degrees = dict(G.degree())
    nodes_to_remove = [node for node, degree in degrees.items() if degree < min_degree]
    G.remove_nodes_from(nodes_to_remove)

    # Layout berechnen (Fruchterman-Reingold wie in R)
    pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)

    # Plot erstellen
    plt.figure(figsize=figsize)

    # Knoten-Farben: Zielterm blau, andere orange
    node_colors = [
        "cornflowerblue" if node == cooc_term else "orange" for node in G.nodes()
    ]

    # Knoten-Größen basierend auf Degree
    degrees = dict(G.degree())
    node_sizes = [300 + 100 * np.log1p(degrees[node]) for node in G.nodes()]

    # Kanten-Breiten basierend auf Signifikanz
    edge_weights = [G[u][v]["sig"] for u, v in G.edges()]
    if edge_weights:
        min_weight = min(edge_weights)
        max_weight = max(edge_weights)
        edge_widths = [
            1 + 9 * (w - min_weight) / (max_weight - min_weight + 0.001)
            for w in edge_weights
        ]
    else:
        edge_widths = [1] * len(G.edges())

    # Zeichne Netzwerk
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.3, edge_color="gray")

    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.9,
        edgecolors="darkgray",
        linewidths=1.5,
    )

    nx.draw_networkx_labels(
        G, pos, font_size=10, font_weight="bold", font_family="sans-serif"
    )

    plt.title(
        f"{cooc_term} Co-Occurrence Graph", fontsize=16, fontweight="bold", pad=20
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")


# ============================================================================
# 7. HAUPTFUNKTION - KOMPLETT-WORKFLOW
# ============================================================================


def analyze_cooccurrences(
    all_documents,
    target_term,
    min_df=10,
    num_coocs=15,
    measure="loglik",
    min_degree=2,
    save_path="",
):
    """
    Kompletter Workflow für Co-Occurrence Analyse

    Parameters:
    -----------
    all_documents : list
        Liste von Dokumenten (Texten)
    target_term : str
        Zielterm für die Analyse
    min_df : int
        Minimale Dokumentfrequenz
    num_coocs : int
        Anzahl der Top Co-Occurrences
    measure : str
        'loglik', 'dice', oder 'mi'
    min_degree : int
        Minimale Anzahl Verbindungen für Visualisierung

    Returns:
    --------
    result_graph : pd.DataFrame
        Netzwerk-Daten
    """
    print("=" * 60)
    print("CO-OCCURRENCE ANALYSE")
    print("=" * 60)

    # 1. Satz-Segmentierung
    print("\n1. Satz-Segmentierung...")
    sentences = split_into_sentences(all_documents)

    # 2. Binäre DTM erstellen
    print("\n2. Erstelle binäre Document-Term-Matrix...")
    bin_dtm, feature_names = create_binary_dtm(
        sentences, min_df=min_df, stop_words="english"
    )

    # 3. Co-Occurrence Matrix berechnen
    print("\n3. Berechne Co-Occurrence Matrix...")
    cooc_matrix = compute_cooccurrence_matrix(bin_dtm)

    # 4. Netzwerk erstellen
    print(f"\n4. Erstelle Netzwerk für Term '{target_term}'...")
    result_graph = build_cooccurrence_network(
        target_term,
        bin_dtm,
        feature_names,
        cooc_matrix,
        num_coocs=num_coocs,
        measure=measure,
    )

    print(f"   Anzahl Kanten: {len(result_graph)}")

    # 5. Top Co-Occurrences anzeigen
    print(f"\n5. Top {num_coocs} Co-Occurrences für '{target_term}':")
    top_coocs = result_graph[result_graph["from"] == target_term].head(num_coocs)
    for _, row in top_coocs.iterrows():
        print(f"   {row['to']:30s} {row['sig']:10.2f}")

    # 6. Visualisierung
    print("\n6. Visualisiere Netzwerk...")
    visualize_cooccurrence_network(
        result_graph, target_term, min_degree=min_degree, save_path=save_path
    )

    return result_graph, top_coocs


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
all_documents = get_documents_for_speeches(environment_speeches)

result_graph, top_coocs = analyze_cooccurrences(
    all_documents,
    target_term="climate",  # Oder 'carbon', 'emission', etc.
    min_df=5,  # Mindestfrequenz
    num_coocs=15,  # Anzahl Co-Occurrences
    measure="loglik",  # Signifikanz-Maß
    min_degree=2,  # Min. Verbindungen für Visualisierung
    save_path=f"{OUTPUT_DIR}/coocc_all.png",
)

top_coocs.to_csv(f"{OUTPUT_DIR}/coocc_all.csv", index_label="ID")

env_cleaned = environment_speeches.copy()
env_cleaned = env_cleaned[env_cleaned["Speaker_birth"] != "-"]
env_cleaned["Speaker_birth"] = env_cleaned["Speaker_birth"].astype(int)
min_y = env_cleaned["Speaker_birth"].min()
max_y = env_cleaned["Speaker_birth"].max()
bins = np.arange(min_y, max_y + 16, 15)
env_cleaned["Birth_Bin"] = pd.cut(env_cleaned["Speaker_birth"], bins=bins, right=False)

# Alle Alterskategorien durchgehen
for birth_bin in env_cleaned["Birth_Bin"].cat.categories:
    print(f"\n{'='*60}")
    print(f"Analysiere Altersgruppe: {birth_bin}")
    print(f"{'='*60}\n")

    # Daten für diese Altersgruppe filtern
    filtered_data = env_cleaned[env_cleaned["Birth_Bin"] == birth_bin]

    print(f"Anzahl Reden in dieser Gruppe: {len(filtered_data)}\n")

    # Analyse durchführen
    result_graph, top_coocs = analyze_cooccurrences(
        get_documents_for_speeches(filtered_data),  # Gefilterte Daten
        target_term="climate",  # Oder 'carbon', 'emission', etc.
        min_df=5,  # Mindestfrequenz
        num_coocs=15,  # Anzahl Co-Occurrences
        measure="loglik",  # Signifikanz-Maß
        min_degree=2,  # Min. Verbindungen für Visualisierung
        save_path=f"{OUTPUT_DIR}/coocc_{birth_bin}.png",
    )
    top_coocs.to_csv(f"{OUTPUT_DIR}/coocc_{birth_bin}.csv", index_label="ID")
