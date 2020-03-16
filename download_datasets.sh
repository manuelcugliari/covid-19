#!/bin/bash

set -e

mkdir -p data/raw/italy
cd data/raw/italy

# Download new italy (or update) dataset here
rm -f dpc-covid19-ita-regioni.csv
rm -f dpc-covid19-ita-andamento-nazionale.csv
wget https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-regioni/dpc-covid19-ita-regioni.csv
wget https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-andamento-nazionale/dpc-covid19-ita-andamento-nazionale.csv

cd ../../../
mkdir -p data/raw/world
cd data/raw/world

# Download new world (or update) dataset here