#!/bin/bash

# ParlaMint und Open Discourse Download und Entpack-Skript
# Speicherort: data/download.sh

set -e  # Beende bei Fehlern

# Sicherstellen, dass wir im richtigen Verzeichnis sind
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ParlaMint Basis-URL und Dateiliste
PARLAMINT_BASE_URL="https://www.clarin.si/repository/xmlui/bitstream/handle/11356/2006"
PARLAMINT_FILES=(
    "ParlaMint-AT-en.ana.tgz"
    "ParlaMint-BA-en.ana.tgz"
    "ParlaMint-BE-en.ana.tgz"
    "ParlaMint-BG-en.ana.tgz"
    "ParlaMint-CZ-en.ana.tgz"
    "ParlaMint-DK-en.ana.tgz"
    "ParlaMint-EE-en.ana.tgz"
    "ParlaMint-ES-en.ana.tgz"
    "ParlaMint-ES-CT-en.ana.tgz"
    "ParlaMint-ES-GA-en.ana.tgz"
    "ParlaMint-ES-PV-en.ana.tgz"
    "ParlaMint-FI-en.ana.tgz"
    "ParlaMint-FR-en.ana.tgz"
    "ParlaMint-GB-en.ana.tgz"
    "ParlaMint-GR-en.ana.tgz"
    "ParlaMint-HR-en.ana.tgz"
    "ParlaMint-HU-en.ana.tgz"
    "ParlaMint-IS-en.ana.tgz"
    "ParlaMint-IT-en.ana.tgz"
    "ParlaMint-LV-en.ana.tgz"
    "ParlaMint-NL-en.ana.tgz"
    "ParlaMint-NO-en.ana.tgz"
    "ParlaMint-PL-en.ana.tgz"
    "ParlaMint-PT-en.ana.tgz"
    "ParlaMint-RS-en.ana.tgz"
    "ParlaMint-SE-en.ana.tgz"
    "ParlaMint-SI-en.ana.tgz"
    "ParlaMint-TR-en.ana.tgz"
    "ParlaMint-UA-en.ana.tgz"
    "ParlaMint-en-logs.tgz"
    "ParlaMint-5.0.tgz"
)

# Open Discourse Dataset Download-URL
# Harvard Dataverse Bundle Download für den gesamten Datensatz
OPEN_DISCOURSE_DOI="doi:10.7910/DVN/FIKIBO"
OPEN_DISCOURSE_URL="https://dataverse.harvard.edu/api/access/dataset/:persistentId/?persistentId=${OPEN_DISCOURSE_DOI}"
OPEN_DISCOURSE_ZIP="open-discourse-dataset.zip"

echo "========================================="
echo "ParlaMint und Open Discourse Download"
echo "========================================="
echo ""

# Phase 1: ParlaMint Download
echo "Phase 1: Download der ParlaMint-Dateien"
echo "----------------------------------------"
echo ""

TOTAL_FILES=${#PARLAMINT_FILES[@]}
DOWNLOADED=0
SKIPPED=0
FAILED=0

for FILE in "${PARLAMINT_FILES[@]}"; do
    URL="${PARLAMINT_BASE_URL}/${FILE}"
    
    if [ -f "$FILE" ]; then
        echo "⏭  Überspringe $FILE (bereits vorhanden)"
        ((SKIPPED++))
    else
        echo "⬇  Lade herunter: $FILE"
        if curl --fail --silent --show-error --remote-name --location "$URL"; then
            echo "✓  Erfolgreich: $FILE"
            ((DOWNLOADED++))
        else
            echo "✗  Fehler beim Download: $FILE"
            ((FAILED++))
        fi
    fi
    echo ""
done

echo "ParlaMint Download-Zusammenfassung:"
echo "  Heruntergeladen: $DOWNLOADED"
echo "  Übersprungen:    $SKIPPED"
echo "  Fehlgeschlagen:  $FAILED"
echo "  Gesamt:          $TOTAL_FILES"
echo ""

# Phase 2: Open Discourse Download
echo "Phase 2: Download des Open Discourse Datensatzes"
echo "-------------------------------------------------"
echo ""

if [ -f "$OPEN_DISCOURSE_ZIP" ]; then
    echo "⏭  Überspringe Open Discourse (bereits vorhanden: $OPEN_DISCOURSE_ZIP)"
    echo ""
else
    echo "⬇  Lade herunter: Open Discourse Dataset (Bundestag Korpus)"
    echo "   DOI: $OPEN_DISCOURSE_DOI"
    echo "   Dies kann einige Minuten dauern, da der Datensatz groß ist..."
    echo ""
    
    if curl --fail --location --output "$OPEN_DISCOURSE_ZIP" "$OPEN_DISCOURSE_URL"; then
        echo "✓  Erfolgreich: Open Discourse Dataset heruntergeladen"
        echo ""
    else
        echo "✗  Fehler beim Download des Open Discourse Datasets"
        echo "   Hinweis: Möglicherweise benötigt der Download einen API-Token oder"
        echo "   es gibt Einschränkungen. Besuche die Dataverse-Seite manuell:"
        echo "   https://dataverse.harvard.edu/dataset.xhtml?persistentId=$OPEN_DISCOURSE_DOI"
        echo ""
        ((FAILED++))
    fi
fi

if [ $FAILED -gt 0 ]; then
    echo "⚠  Warnung: Einige Downloads sind fehlgeschlagen."
    echo "   Führe das Skript erneut aus, um fehlende Dateien herunterzuladen."
    echo ""
fi

# Phase 3: ParlaMint Entpacken
echo "Phase 3: Entpacken der ParlaMint Archive"
echo "-----------------------------------------"
echo ""

EXTRACTED=0
SKIPPED_EXTRACT=0
FAILED_EXTRACT=0

for FILE in "${PARLAMINT_FILES[@]}"; do
    if [ ! -f "$FILE" ]; then
        echo "⏭  Überspringe $FILE (Datei nicht vorhanden)"
        ((SKIPPED_EXTRACT++))
        continue
    fi
    
    # Verzeichnisname ohne .tgz-Endung
    DIR_NAME="${FILE%.tgz}"
    
    if [ -d "$DIR_NAME" ]; then
        echo "⏭  Überspringe $FILE (bereits entpackt nach $DIR_NAME)"
        ((SKIPPED_EXTRACT++))
    else
        echo "📦 Entpacke: $FILE"
        if tar -xzf "$FILE"; then
            echo "✓  Erfolgreich entpackt: $FILE"
            ((EXTRACTED++))
        else
            echo "✗  Fehler beim Entpacken: $FILE"
            ((FAILED_EXTRACT++))
        fi
    fi
    echo ""
done

echo "ParlaMint Entpack-Zusammenfassung:"
echo "  Entpackt:        $EXTRACTED"
echo "  Übersprungen:    $SKIPPED_EXTRACT"
echo "  Fehlgeschlagen:  $FAILED_EXTRACT"
echo "  Gesamt:          $TOTAL_FILES"
echo ""

# Phase 4: Open Discourse Entpacken
echo "Phase 4: Entpacken des Open Discourse Datasets"
echo "-----------------------------------------------"
echo ""

if [ ! -f "$OPEN_DISCOURSE_ZIP" ]; then
    echo "⏭  Überspringe Open Discourse (ZIP-Datei nicht vorhanden)"
    echo ""
else
    OPEN_DISCOURSE_DIR="open-discourse"
    
    if [ -d "$OPEN_DISCOURSE_DIR" ]; then
        echo "⏭  Überspringe Open Discourse (bereits entpackt nach $OPEN_DISCOURSE_DIR)"
        echo ""
    else
        echo "📦 Entpacke: $OPEN_DISCOURSE_ZIP"
        mkdir -p "$OPEN_DISCOURSE_DIR"
        if unzip -q "$OPEN_DISCOURSE_ZIP" -d "$OPEN_DISCOURSE_DIR"; then
            echo "✓  Erfolgreich entpackt: Open Discourse Dataset"
            echo "   Verzeichnis: $OPEN_DISCOURSE_DIR"
            echo ""
        else
            echo "✗  Fehler beim Entpacken von $OPEN_DISCOURSE_ZIP"
            echo ""
            ((FAILED_EXTRACT++))
        fi
    fi
fi

# Abschluss
echo "========================================="
if [ $FAILED -eq 0 ] && [ $FAILED_EXTRACT -eq 0 ]; then
    echo "✓  Alle Operationen erfolgreich abgeschlossen!"
    echo ""
    echo "Heruntergeladene Datensätze:"
    echo "  - ParlaMint: Parlamentsprotokolle aus 31 europäischen Ländern"
    echo "  - Open Discourse: Deutsche Bundestagsprotokolle (1949-heute)"
else
    echo "⚠  Einige Operationen sind fehlgeschlagen."
    echo "   Führe das Skript erneut aus, um fortzufahren."
fi
echo "========================================="
