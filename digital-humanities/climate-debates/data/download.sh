#!/bin/bash

# ParlaMint Download und Entpack-Skript
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

echo "========================================="
echo "ParlaMint Download"
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

# Phase 2: ParlaMint Entpacken
echo "Phase 2: Entpacken der ParlaMint Archive"
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
        echo "📦 Entpacke: $FILE nach $DIR_NAME/"
        mkdir -p "$DIR_NAME"
        if tar -xzf "$FILE" -C "$DIR_NAME"; then
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

# Abschluss
echo "========================================="
if [ $FAILED -eq 0 ] && [ $FAILED_EXTRACT -eq 0 ]; then
    echo "✓  Alle Operationen erfolgreich abgeschlossen!"
    echo ""
    echo "Heruntergeladene Datensätze:"
    echo "  - ParlaMint: Parlamentsprotokolle aus 29 europäischen Ländern"
else
    echo "⚠  Einige Operationen sind fehlgeschlagen."
    echo "   Führe das Skript erneut aus, um fortzufahren."
fi
echo "========================================="
