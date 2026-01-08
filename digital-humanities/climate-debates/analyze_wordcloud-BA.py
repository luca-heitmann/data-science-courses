#!/usr/bin/env python3
"""
ParlaMint Klimawandel Word Cloud Analyse
Erstellt verschiedene Word Clouds für Klimawandel-Debatten
"""

import os
import re
from pathlib import Path
from xml.etree import ElementTree as ET
from typing import List, Dict
import pandas as pd
import numpy as np
from collections import Counter, defaultdict

# NLP
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Word Cloud
from wordcloud import WordCloud

# Visualisierung
import matplotlib.pyplot as plt
import seaborn as sns

# Farbpaletten
from matplotlib.colors import LinearSegmentedColormap


class ParlaMintWordCloudAnalyzer:
    """Erstellt Word Clouds für ParlaMint Klimawandel-Debatten"""

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

    NS = {
        "tei": "http://www.tei-c.org/ns/1.0",
        "xml": "http://www.w3.org/XML/1998/namespace",
    }

    def __init__(self, data_dir: str = "data"):
        """Initialisiert den Analyzer"""
        self.data_dir = Path(data_dir)
        self.corpus_dir = self.data_dir / "ParlaMint-BA-en.TEI.ana"
        self.speeches = []
        self.climate_speeches = []

        # Erweiterte Stopwords für bessere Word Clouds
        try:
            self.stop_words = set(stopwords.words("english"))
        except LookupError:
            nltk.download("stopwords")
            nltk.download("punkt")
            self.stop_words = set(stopwords.words("english"))

        # Zusätzliche Stopwords für parlamentarische Debatten
        self.stop_words.update(
            [
                "mr",
                "mrs",
                "ms",
                "madam",
                "sir",
                "president",
                "chairman",
                "speaker",
                "member",
                "members",
                "colleague",
                "colleagues",
                "hon",
                "honorable",
                "house",
                "parliament",
                "chamber",
                "session",
                "debate",
                "said",
                "says",
                "thank",
                "thanks",
                "would",
                "could",
                "also",
                "one",
                "two",
                "three",
                "like",
                "want",
                "need",
                "get",
                "know",
                "think",
                "make",
                "time",
                "year",
                "years",
                "question",
                "questions",
            ]
        )

    def extract_text_from_xml(self, xml_file: Path) -> List[Dict]:
        """Extrahiert Reden aus ParlaMint XML-Dateien"""
        speeches = []

        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()

            date_elem = root.find(".//tei:setting/tei:date", self.NS)
            session_date = date_elem.get("when") if date_elem is not None else "unknown"

            for u in root.findall(".//tei:u", self.NS):
                speech_id = u.get(f'{{{self.NS["xml"]}}}id', "unknown")
                speaker_id = u.get("who", "unknown")

                segments = []
                for seg in u.findall(".//tei:seg", self.NS):
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
        """Lädt alle XML-Dateien aus dem Korpus"""
        print(f"Lade Korpus aus: {self.corpus_dir}")

        if not self.corpus_dir.exists():
            raise FileNotFoundError(
                f"Korpus-Verzeichnis nicht gefunden: {self.corpus_dir}\n"
                f"Stelle sicher, dass data/download.sh ausgeführt wurde."
            )

        xml_files = list(self.corpus_dir.rglob("*.xml"))
        xml_files = [
            f
            for f in xml_files
            if not f.name.endswith(".ana.xml") or "ParlaMint-BA-en" in f.name
        ]

        print(f"Gefundene XML-Dateien: {len(xml_files)}")

        if year_filter:
            xml_files = [
                f for f in xml_files if any(str(year) in f.name for year in year_filter)
            ]
            print(f"Nach Jahr-Filter: {len(xml_files)} Dateien")

        for xml_file in xml_files:
            speeches = self.extract_text_from_xml(xml_file)
            self.speeches.extend(speeches)

            if len(self.speeches) % 100 == 0:
                print(f"Verarbeitet: {len(self.speeches)} Reden...")

        print(f"\nGesamt extrahierte Reden: {len(self.speeches)}")

    def filter_climate_speeches(self):
        """Filtert Reden mit Klimawandel-Bezug"""
        print("\nFiltere Klimawandel-relevante Reden...")

        for speech in self.speeches:
            text_lower = speech["text"].lower()

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

    def preprocess_text(self, text: str, remove_stopwords: bool = True) -> str:
        """
        Bereinigt Text für Word Cloud

        Args:
            text: Zu bereinigender Text
            remove_stopwords: Ob Stopwords entfernt werden sollen

        Returns:
            Bereinigter Text
        """
        text = text.lower()
        text = re.sub(r"[^a-z\s]", " ", text)
        text = re.sub(r"\s+", " ", text)

        if remove_stopwords:
            words = text.split()
            words = [w for w in words if w not in self.stop_words and len(w) > 2]
            text = " ".join(words)

        return text.strip()

    def get_word_frequencies(
        self, texts: List[str], top_n: int = None
    ) -> Dict[str, int]:
        """
        Berechnet Worthäufigkeiten

        Args:
            texts: Liste von Texten
            top_n: Optional, nur top N Wörter zurückgeben

        Returns:
            Dictionary mit Worthäufigkeiten
        """
        all_words = []
        for text in texts:
            cleaned = self.preprocess_text(text)
            all_words.extend(cleaned.split())

        word_freq = Counter(all_words)

        if top_n:
            word_freq = dict(word_freq.most_common(top_n))

        return word_freq

    def create_basic_wordcloud(
        self,
        word_freq: Dict[str, int],
        title: str = "Word Cloud",
        filename: str = "wordcloud.png",
        colormap: str = "viridis",
    ):
        """
        Erstellt eine einfache Word Cloud

        Args:
            word_freq: Dictionary mit Worthäufigkeiten
            title: Titel der Visualisierung
            filename: Dateiname für Speicherung
            colormap: Matplotlib colormap
        """
        wordcloud = WordCloud(
            width=1600,
            height=800,
            background_color="white",
            colormap=colormap,
            relative_scaling=0.5,
            min_font_size=10,
            max_words=200,
        ).generate_from_frequencies(word_freq)

        plt.figure(figsize=(20, 10))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title(title, fontsize=24, pad=20)
        plt.tight_layout(pad=0)

        output_path = self.data_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Word Cloud gespeichert: {output_path}")
        plt.close()

    def create_comparative_wordcloud(self, years: List[int] = None):
        """
        Erstellt vergleichende Word Clouds für verschiedene Zeiträume

        Args:
            years: Liste von Jahren zum Vergleichen
        """
        if not self.climate_speeches:
            print("Keine Klimawandel-Reden gefunden!")
            return

        if not years or len(years) < 2:
            print("Mindestens 2 Jahre für Vergleich benötigt!")
            return

        fig, axes = plt.subplots(1, len(years), figsize=(10 * len(years), 8))

        if len(years) == 2:
            axes = [axes[0], axes[1]]

        for idx, year in enumerate(years):
            # Reden für dieses Jahr filtern
            year_speeches = [
                s for s in self.climate_speeches if str(year) in s.get("date", "")
            ]

            if not year_speeches:
                print(f"Keine Reden für Jahr {year} gefunden!")
                continue

            # Texte zusammenführen
            texts = [s["text"] for s in year_speeches]
            word_freq = self.get_word_frequencies(texts, top_n=100)

            # Word Cloud erstellen
            wordcloud = WordCloud(
                width=800,
                height=600,
                background_color="white",
                colormap="Spectral",
                relative_scaling=0.5,
                min_font_size=8,
            ).generate_from_frequencies(word_freq)

            # Plotten
            axes[idx].imshow(wordcloud, interpolation="bilinear")
            axes[idx].axis("off")
            axes[idx].set_title(
                f"Jahr {year}\n({len(year_speeches)} Reden)", fontsize=16
            )

        plt.tight_layout()
        output_path = self.data_dir / "wordcloud_comparative.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Vergleichende Word Cloud gespeichert: {output_path}")
        plt.close()

    def create_keyword_context_wordcloud(self, keyword: str = "climate change"):
        """
        Erstellt Word Cloud für Kontext um ein bestimmtes Keyword

        Args:
            keyword: Keyword, um das der Kontext analysiert wird
        """
        if not self.climate_speeches:
            print("Keine Klimawandel-Reden gefunden!")
            return

        print(f"\nErstelle Kontext-Word Cloud für: '{keyword}'")

        # Extrahiere Kontext um Keyword (±100 Wörter)
        context_texts = []
        for speech in self.climate_speeches:
            text = speech["text"].lower()
            if keyword.lower() in text:
                # Finde alle Positionen des Keywords
                words = text.split()
                for i, word in enumerate(words):
                    if keyword.lower() in word:
                        # Extrahiere Kontext
                        start = max(0, i - 50)
                        end = min(len(words), i + 50)
                        context = " ".join(words[start:end])
                        context_texts.append(context)

        if not context_texts:
            print(f"Kein Kontext für '{keyword}' gefunden!")
            return

        # Word Frequencies berechnen (exkludiere das Keyword selbst)
        word_freq = self.get_word_frequencies(context_texts, top_n=150)

        # Entferne das Keyword selbst und seine Bestandteile
        keyword_parts = keyword.lower().split()
        for part in keyword_parts:
            word_freq.pop(part, None)

        # Word Cloud erstellen
        safe_keyword = keyword.replace(" ", "_")
        self.create_basic_wordcloud(
            word_freq,
            title=f'Kontext-Wörter um "{keyword}"',
            filename=f"wordcloud_context_{safe_keyword}.png",
            colormap="RdYlGn",
        )

    def create_bigram_wordcloud(self):
        """Erstellt Word Cloud aus Bigrammen (Zwei-Wort-Phrasen)"""
        if not self.climate_speeches:
            print("Keine Klimawandel-Reden gefunden!")
            return

        print("\nErstelle Bigram Word Cloud...")

        # Bigrams extrahieren
        bigram_freq = Counter()
        for speech in self.climate_speeches:
            text = self.preprocess_text(speech["text"])
            words = text.split()

            for i in range(len(words) - 1):
                bigram = f"{words[i]} {words[i+1]}"
                if len(words[i]) > 2 and len(words[i + 1]) > 2:
                    bigram_freq[bigram] += 1

        # Top Bigrams
        top_bigrams = dict(bigram_freq.most_common(100))

        self.create_basic_wordcloud(
            top_bigrams,
            title="Häufigste Zwei-Wort-Phrasen in Klimawandel-Debatten",
            filename="wordcloud_bigrams.png",
            colormap="plasma",
        )

        # Top 20 Bigrams ausgeben
        print("\nTop 20 Bigrams:")
        for bigram, count in bigram_freq.most_common(20):
            print(f"  {bigram}: {count}")

    def create_speaker_wordcloud(self, top_speakers: int = 5):
        """
        Erstellt separate Word Clouds für Top-Sprecher

        Args:
            top_speakers: Anzahl der Top-Sprecher
        """
        if not self.climate_speeches:
            print("Keine Klimawandel-Reden gefunden!")
            return

        print(f"\nErstelle Word Clouds für Top {top_speakers} Sprecher...")

        # Zähle Reden pro Sprecher
        speaker_counts = Counter()
        speaker_texts = defaultdict(list)

        for speech in self.climate_speeches:
            speaker = speech.get("speaker", "unknown")
            speaker_counts[speaker] += 1
            speaker_texts[speaker].append(speech["text"])

        # Top Sprecher
        top_speaker_ids = [
            s[0] for s in speaker_counts.most_common(top_speakers) if s[0] != "unknown"
        ]

        if not top_speaker_ids:
            print("Keine Sprecher-Informationen verfügbar!")
            return

        # Für jeden Top-Sprecher eine Word Cloud
        for idx, speaker_id in enumerate(top_speaker_ids):
            texts = speaker_texts[speaker_id]
            word_freq = self.get_word_frequencies(texts, top_n=100)

            # Bereinige Sprecher-ID für Dateiname
            safe_speaker = re.sub(r"[^a-zA-Z0-9]", "_", speaker_id)

            self.create_basic_wordcloud(
                word_freq,
                title=f"Sprecher: {speaker_id} ({len(texts)} Reden)",
                filename=f"wordcloud_speaker_{idx+1}_{safe_speaker}.png",
                colormap="coolwarm",
            )

    def create_frequency_analysis(self, top_n: int = 30):
        """
        Erstellt Balkendiagramm mit häufigsten Wörtern

        Args:
            top_n: Anzahl der häufigsten Wörter
        """
        if not self.climate_speeches:
            print("Keine Klimawandel-Reden gefunden!")
            return

        print(f"\nErstelle Häufigkeitsanalyse (Top {top_n} Wörter)...")

        texts = [s["text"] for s in self.climate_speeches]
        word_freq = self.get_word_frequencies(texts, top_n=top_n)

        # Sortiere nach Häufigkeit
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        words, counts = zip(*sorted_words)

        # Visualisierung
        fig, ax = plt.subplots(figsize=(14, 10))
        bars = ax.barh(
            range(len(words)),
            counts,
            color=plt.cm.viridis(np.linspace(0, 1, len(words))),
        )
        ax.set_yticks(range(len(words)))
        ax.set_yticklabels(words)
        ax.invert_yaxis()
        ax.set_xlabel("Häufigkeit", fontsize=12)
        ax.set_title(
            f"Top {top_n} häufigste Wörter in Klimawandel-Debatten", fontsize=14
        )
        ax.grid(axis="x", alpha=0.3)

        plt.tight_layout()
        output_path = self.data_dir / "word_frequency_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Häufigkeitsanalyse gespeichert: {output_path}")
        plt.close()

    def create_all_wordclouds(self, years: List[int] = None):
        """
        Erstellt alle Word Cloud-Typen

        Args:
            years: Jahre für vergleichende Analyse
        """
        if not self.climate_speeches:
            print("Keine Klimawandel-Reden gefunden!")
            return

        print("\n" + "=" * 80)
        print("Erstelle alle Word Clouds...")
        print("=" * 80)

        # 1. Haupt-Word Cloud
        print("\n1. Haupt-Word Cloud")
        texts = [s["text"] for s in self.climate_speeches]
        word_freq = self.get_word_frequencies(texts, top_n=200)
        self.create_basic_wordcloud(
            word_freq,
            title="Klimawandel-Debatten: Häufigste Begriffe",
            filename="wordcloud_main.png",
            colormap="viridis",
        )

        # 2. Bigram Word Cloud
        print("\n2. Bigram Word Cloud")
        self.create_bigram_wordcloud()

        # 3. Kontext Word Clouds für verschiedene Keywords
        print("\n3. Kontext Word Clouds")
        context_keywords = ["climate change", "renewable energy", "fossil fuel"]
        for keyword in context_keywords:
            self.create_keyword_context_wordcloud(keyword)

        # 4. Vergleichende Word Clouds (falls Jahre angegeben)
        if years and len(years) >= 2:
            print("\n4. Vergleichende Word Clouds")
            self.create_comparative_wordcloud(years)

        # 5. Sprecher Word Clouds
        print("\n5. Sprecher Word Clouds")
        self.create_speaker_wordcloud(top_speakers=3)

        # 6. Häufigkeitsanalyse
        print("\n6. Häufigkeitsanalyse")
        self.create_frequency_analysis(top_n=30)

        print("\n" + "=" * 80)
        print("Alle Word Clouds erstellt!")
        print("=" * 80)


def main():
    """Hauptfunktion"""
    print("=" * 80)
    print("ParlaMint Klimawandel Word Cloud Analyse")
    print("=" * 80)

    # Analyzer initialisieren
    analyzer = ParlaMintWordCloudAnalyzer(data_dir="data")

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

    # Alle Word Clouds erstellen
    analyzer.create_all_wordclouds(years=[2021, 2022])

    print("\n" + "=" * 80)
    print("Analyse abgeschlossen!")
    print("Alle Word Clouds wurden im data/ Verzeichnis gespeichert.")
    print("=" * 80)


if __name__ == "__main__":
    main()
