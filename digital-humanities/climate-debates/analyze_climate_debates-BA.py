#!/usr/bin/env python3
"""
ParlaMint Klimawandel Analyse mit Topic Modelling
Analysiert ParlaMint-BA-en.ana Korpus für Klimawandel-Debatten
"""

import os
import re
from pathlib import Path
from xml.etree import ElementTree as ET
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
from collections import defaultdict

# NLP und Topic Modelling
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Visualisierung
import matplotlib.pyplot as plt
import seaborn as sns

# Für BERTopic (optional, falls installiert)
try:
    from bertopic import BERTopic

    BERTOPIC_AVAILABLE = True
except ImportError:
    BERTOPIC_AVAILABLE = False
    print("Hinweis: BERTopic nicht installiert. Verwende klassische Topic Modelle.")


class ParlaMintClimateAnalyzer:
    """Analysiert ParlaMint Korpora für Klimawandel-Debatten"""

    # Klimawandel-Schlüsselwörter (erweiterte Liste)
    CLIMATE_KEYWORDS = [
        "climate change",
        "global warming",
        "greenhouse gas",
        "carbon emission",
        "carbon dioxide",
        "co2",
        "climate crisis",
        "climate emergency",
        "renewable energy",
        "fossil fuel",
        "sustainability",
        "carbon footprint",
        "climate policy",
        "paris agreement",
        "climate action",
        "net zero",
        "decarbonization",
        "climate adaptation",
        "climate mitigation",
        "environmental protection",
        "green economy",
        "climate justice",
    ]

    # TEI Namespaces
    NS = {
        "tei": "http://www.tei-c.org/ns/1.0",
        "xml": "http://www.w3.org/XML/1998/namespace",
    }

    def __init__(self, data_dir: str = "data"):
        """
        Initialisiert den Analyzer

        Args:
            data_dir: Pfad zum data-Verzeichnis
        """
        self.data_dir = Path(data_dir)
        self.corpus_dir = self.data_dir / "ParlaMint-BA-en.TEI.ana"
        self.speeches = []
        self.climate_speeches = []

        # NLTK Daten herunterladen
        try:
            stopwords.words("english")
        except LookupError:
            nltk.download("stopwords")
            nltk.download("punkt")

    def extract_text_from_xml(self, xml_file: Path) -> List[Dict]:
        """
        Extrahiert Reden aus ParlaMint XML-Dateien

        Args:
            xml_file: Pfad zur XML-Datei

        Returns:
            Liste von Dictionaries mit Speech-Informationen
        """
        speeches = []

        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()

            # Datum der Sitzung extrahieren
            date_elem = root.find(".//tei:setting/tei:date", self.NS)
            session_date = date_elem.get("when") if date_elem is not None else "unknown"

            # Alle Reden (utterances) finden
            for u in root.findall(".//tei:u", self.NS):
                speech_id = u.get(f'{{{self.NS["xml"]}}}id', "unknown")
                speaker_id = u.get("who", "unknown")

                # Text aus allen Segmenten extrahieren
                segments = []
                for seg in u.findall(".//tei:seg", self.NS):
                    # Text aus Wörtern rekonstruieren
                    words = []
                    for w in seg.findall(".//tei:w", self.NS):
                        words.append(w.text or "")
                    for pc in seg.findall(".//tei:pc", self.NS):
                        words.append(pc.text or "")

                    if words:
                        segments.append(" ".join(words))

                if segments:
                    full_text = " ".join(segments)

                    speeches.append(
                        {
                            "id": speech_id,
                            "speaker": speaker_id,
                            "date": session_date,
                            "text": full_text,
                            "file": xml_file.name,
                            "text_length": len(full_text),
                        }
                    )

        except Exception as e:
            print(f"Fehler beim Parsen von {xml_file}: {e}")

        return speeches

    def load_corpus(self, year_filter: List[int] = None):
        """
        Lädt alle XML-Dateien aus dem Korpus

        Args:
            year_filter: Optional, Liste von Jahren zum Filtern
        """
        print(f"Lade Korpus aus: {self.corpus_dir}")

        if not self.corpus_dir.exists():
            raise FileNotFoundError(
                f"Korpus-Verzeichnis nicht gefunden: {self.corpus_dir}\n"
                f"Stelle sicher, dass data/download.sh ausgeführt wurde."
            )

        # Alle XML-Dateien finden
        xml_files = list(self.corpus_dir.rglob("*.xml"))
        xml_files = [
            f
            for f in xml_files
            if not f.name.endswith(".ana.xml") or "ParlaMint-BA-en" in f.name
        ]

        print(f"Gefundene XML-Dateien: {len(xml_files)}")

        # Jahr-Filter anwenden
        if year_filter:
            xml_files = [
                f for f in xml_files if any(str(year) in f.name for year in year_filter)
            ]
            print(f"Nach Jahr-Filter: {len(xml_files)} Dateien")

        # Reden extrahieren
        for xml_file in xml_files:
            speeches = self.extract_text_from_xml(xml_file)
            self.speeches.extend(speeches)

            if len(self.speeches) % 100 == 0:
                print(f"Verarbeitet: {len(self.speeches)} Reden...")

        print(f"\nGesamt extrahierte Reden: {len(self.speeches)}")

    def filter_climate_speeches(self, context_window: int = 200):
        """
        Filtert Reden mit Klimawandel-Bezug

        Args:
            context_window: Anzahl Zeichen um Keywords für Kontext
        """
        print("\nFiltere Klimawandel-relevante Reden...")

        for speech in self.speeches:
            text_lower = speech["text"].lower()

            # Check für Keywords
            matches = []
            for keyword in self.CLIMATE_KEYWORDS:
                if keyword in text_lower:
                    matches.append(keyword)

            if matches:
                speech_copy = speech.copy()
                speech_copy["matched_keywords"] = matches
                speech_copy["keyword_count"] = len(matches)
                self.climate_speeches.append(speech_copy)

        print(f"Gefundene Klimawandel-Reden: {len(self.climate_speeches)}")

        if self.climate_speeches:
            # Statistiken
            keyword_freq = defaultdict(int)
            for speech in self.climate_speeches:
                for kw in speech["matched_keywords"]:
                    keyword_freq[kw] += 1

            print("\nTop 10 Keywords:")
            for kw, count in sorted(
                keyword_freq.items(), key=lambda x: x[1], reverse=True
            )[:10]:
                print(f"  {kw}: {count}")

    def preprocess_text(self, text: str) -> str:
        """
        Bereinigt Text für Topic Modelling

        Args:
            text: Zu bereinigender Text

        Returns:
            Bereinigter Text
        """
        # Lowercase
        text = text.lower()

        # Entferne Sonderzeichen, behalte nur Buchstaben und Leerzeichen
        text = re.sub(r"[^a-z\s]", " ", text)

        # Entferne mehrfache Leerzeichen
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def perform_lda_topic_modeling(self, n_topics: int = 5, max_features: int = 1000):
        """
        Führt LDA Topic Modelling durch

        Args:
            n_topics: Anzahl der Topics
            max_features: Maximale Anzahl Features
        """
        if not self.climate_speeches:
            print("Keine Klimawandel-Reden gefunden!")
            return None, None, None

        print(f"\nFühre LDA Topic Modelling durch ({n_topics} Topics)...")

        # Texte vorbereiten
        texts = [self.preprocess_text(s["text"]) for s in self.climate_speeches]

        # Vectorizer
        stop_words = list(stopwords.words("english"))
        vectorizer = CountVectorizer(
            max_features=max_features, stop_words=stop_words, min_df=2, max_df=0.8
        )

        doc_term_matrix = vectorizer.fit_transform(texts)

        # LDA Model
        lda_model = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            max_iter=50,
            learning_method="online",
        )

        lda_topics = lda_model.fit_transform(doc_term_matrix)

        # Topics anzeigen
        feature_names = vectorizer.get_feature_names_out()
        print("\nExtrahierte Topics:")
        print("=" * 80)

        for topic_idx, topic in enumerate(lda_model.components_):
            top_words_idx = topic.argsort()[-10:][::-1]
            top_words = [feature_names[i] for i in top_words_idx]
            print(f"\nTopic {topic_idx + 1}:")
            print(f"  Top Words: {', '.join(top_words)}")

        return lda_model, vectorizer, lda_topics

    def perform_nmf_topic_modeling(self, n_topics: int = 5, max_features: int = 1000):
        """
        Führt NMF Topic Modelling durch

        Args:
            n_topics: Anzahl der Topics
            max_features: Maximale Anzahl Features
        """
        if not self.climate_speeches:
            print("Keine Klimawandel-Reden gefunden!")
            return None, None, None

        print(f"\nFühre NMF Topic Modelling durch ({n_topics} Topics)...")

        # Texte vorbereiten
        texts = [self.preprocess_text(s["text"]) for s in self.climate_speeches]

        # TF-IDF Vectorizer (besser für NMF)
        stop_words = list(stopwords.words("english"))
        tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features, stop_words=stop_words, min_df=2, max_df=0.8
        )

        tfidf_matrix = tfidf_vectorizer.fit_transform(texts)

        # NMF Model
        nmf_model = NMF(n_components=n_topics, random_state=42, max_iter=200)

        nmf_topics = nmf_model.fit_transform(tfidf_matrix)

        # Topics anzeigen
        feature_names = tfidf_vectorizer.get_feature_names_out()
        print("\nExtrahierte Topics (NMF):")
        print("=" * 80)

        for topic_idx, topic in enumerate(nmf_model.components_):
            top_words_idx = topic.argsort()[-10:][::-1]
            top_words = [feature_names[i] for i in top_words_idx]
            print(f"\nTopic {topic_idx + 1}:")
            print(f"  Top Words: {', '.join(top_words)}")

        return nmf_model, tfidf_vectorizer, nmf_topics

    def visualize_topics(self, model, vectorizer, doc_topics, method: str = "LDA"):
        """
        Visualisiert Topic-Verteilungen

        Args:
            model: Trainiertes Topic Model
            vectorizer: Verwendeter Vectorizer
            doc_topics: Dokument-Topic Matrix
            method: Name der Methode (für Titel)
        """
        # Topic-Verteilung über Dokumente
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 1. Heatmap der Top-Words pro Topic
        feature_names = vectorizer.get_feature_names_out()
        n_top_words = 10

        topic_words = []
        for topic in model.components_:
            top_idx = topic.argsort()[-n_top_words:][::-1]
            topic_words.append([feature_names[i] for i in top_idx])

        # 2. Dokument-Topic Verteilung
        topic_distribution = doc_topics.mean(axis=0)
        axes[0].bar(range(len(topic_distribution)), topic_distribution)
        axes[0].set_xlabel("Topic")
        axes[0].set_ylabel("Durchschnittlicher Anteil")
        axes[0].set_title(f"{method}: Topic-Verteilung über Dokumente")
        axes[0].set_xticks(range(len(topic_distribution)))
        axes[0].set_xticklabels([f"T{i+1}" for i in range(len(topic_distribution))])

        # 3. Top Topics Heatmap
        top_topics_per_doc = doc_topics.argmax(axis=1)
        topic_counts = np.bincount(top_topics_per_doc, minlength=doc_topics.shape[1])

        axes[1].bar(range(len(topic_counts)), topic_counts)
        axes[1].set_xlabel("Topic")
        axes[1].set_ylabel("Anzahl Dokumente")
        axes[1].set_title(f"{method}: Dominante Topics in Dokumenten")
        axes[1].set_xticks(range(len(topic_counts)))
        axes[1].set_xticklabels([f"T{i+1}" for i in range(len(topic_counts))])

        plt.tight_layout()
        plt.savefig(
            self.data_dir / f"topic_analysis_{method.lower()}.png",
            dpi=300,
            bbox_inches="tight",
        )
        print(f"\nVisualisierung gespeichert: topic_analysis_{method.lower()}.png")
        plt.close()

    def export_results(self, lda_model=None, nmf_model=None):
        """Exportiert Ergebnisse als CSV"""
        if self.climate_speeches:
            # DataFrame erstellen
            df = pd.DataFrame(self.climate_speeches)

            # Export
            output_file = self.data_dir / "climate_speeches.csv"
            df.to_csv(output_file, index=False, encoding="utf-8")
            print(f"\nErgebnisse exportiert: {output_file}")

            # Statistiken
            print("\nStatistiken:")
            print(f"  Anzahl Reden: {len(df)}")
            print(f"  Durchschnittliche Länge: {df['text_length'].mean():.0f} Zeichen")
            if "date" in df.columns:
                print(f"  Zeitraum: {df['date'].min()} bis {df['date'].max()}")


def main():
    """Hauptfunktion"""
    print("=" * 80)
    print("ParlaMint Klimawandel Topic Modelling Analyse")
    print("=" * 80)

    # Analyzer initialisieren
    analyzer = ParlaMintClimateAnalyzer(data_dir="data")

    # Korpus laden
    try:
        analyzer.load_corpus(year_filter=[2021, 2022])
    except FileNotFoundError as e:
        print(f"\nFehler: {e}")
        print("\nBitte führe zuerst das Download-Skript aus:")
        print("  ./data/download.sh")
        return

    # Klimawandel-Reden filtern
    analyzer.filter_climate_speeches()

    if not analyzer.climate_speeches:
        print("\nKeine Klimawandel-Reden gefunden!")
        print("Möglicherweise enthält der Korpus keine relevanten Debatten.")
        return

    # Topic Modelling durchführen
    print("\n" + "=" * 80)
    print("Topic Modelling")
    print("=" * 80)

    # LDA
    lda_model, lda_vec, lda_topics = analyzer.perform_lda_topic_modeling(
        n_topics=5, max_features=1000
    )

    if lda_model:
        analyzer.visualize_topics(lda_model, lda_vec, lda_topics, method="LDA")

    # NMF
    nmf_model, nmf_vec, nmf_topics = analyzer.perform_nmf_topic_modeling(
        n_topics=5, max_features=1000
    )

    if nmf_model:
        analyzer.visualize_topics(nmf_model, nmf_vec, nmf_topics, method="NMF")

    # Ergebnisse exportieren
    analyzer.export_results(lda_model, nmf_model)

    print("\n" + "=" * 80)
    print("Analyse abgeschlossen!")
    print("=" * 80)


if __name__ == "__main__":
    main()
