#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Environmental Policy Stringency Index

EPSI_URL="https://sdmx.oecd.org/archive/rest/data/OECD,DF_EPS,/all?dimensionAtObservation=AllDimensions&format=csvfilewithlabels"

curl "$EPSI_URL" -o oecd_eps.csv

