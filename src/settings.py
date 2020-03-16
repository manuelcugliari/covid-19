"""
    File name: settings.py
    Author: Manuel Cugliari
    Date created: 15/03/2020
    Python Version: 3.7.4
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DATA_FOLDER = PROJECT_ROOT / 'data' / 'processed'
MODELS_FOLDER = PROJECT_ROOT / 'models'
LOGS_FOLDER = PROJECT_ROOT / 'tensorboard_logs'

RAW_DATA_FOLDER = PROJECT_ROOT / 'data' / 'raw'

ITALY_RAW_DATA_FOLDER = PROJECT_ROOT / 'data' / 'raw' / 'italy'
ITALY_PROCESSED_DATA_FOLDER = PROJECT_ROOT / 'data' / 'processed' / 'italy'
ITALY_TRAINING_DATA_FOLDER = PROJECT_ROOT / 'data' / 'training' / 'italy'

WORLD_RAW_DATA_FOLDER = PROJECT_ROOT / 'data' / 'raw' / 'world'
WORLD_PROCESSED_DATA_FOLDER = PROJECT_ROOT / 'data' / 'processed' / 'world'
WORLD_TRAINING_DATA_FOLDER = PROJECT_ROOT / 'data' / 'training' / 'world'

sys.path.append(str(PROJECT_ROOT / 'src'))

header = {'ricoverati_con_sintomi': 'HospitalizedPatients',
          'terapia_intensiva': 'IntensiveCarePatients',
          'totale_ospedalizzati': 'TotalHospitalizedPatients',
          'isolamento_domiciliare': 'HomeConfinement',
          'totale_attualmente_positivi': 'CurrentPositiveCases',
          'nuovi_attualmente_positivi': 'NewPositiveCases',
          'dimessi_guariti': 'Recovered',
          'deceduti': 'Deaths',
          'totale_casi': 'TotalPositiveCases',
          'tamponi': 'TestsPerformed'}

region_code = {13: 'Abruzzo',
               17: 'Basilicata',
               4: 'P.A. Bolzano',
               18: 'Calabria',
               15: 'Campania',
               8: 'Emilia Romagna',
               6: 'Friuli Venezia Giulia',
               12: 'Lazio',
               7: 'Liguria',
               3: 'Lombardia',
               11: 'Marche',
               14: 'Molise',
               1: 'Piemonte',
               16: 'Puglia',
               20: 'Sardegna',
               19: 'Sicilia',
               9: 'Toscana',
               4: 'P.A. Trento',
               10: 'Umbria',
               2: 'Valle d\'Aosta',
               5: 'Veneto'}

country_dataset = 'dpc-covid19-ita-andamento-nazionale'
region_dataset = 'dpc-covid19-ita-regioni'

supported_dimensions = ('ricoverati_con_sintomi', 'terapia_intensiva',
                        'totale_ospedalizzati', 'isolamento_domiciliare', 'totale_attualmente_positivi',
                        'nuovi_attualmente_positivi',
                        'dimessi_guariti', 'deceduti', 'totale_casi', 'tamponi')

supported_coutries = ('Italy', 'Hubei')
