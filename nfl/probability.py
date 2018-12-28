# import referencedata
import pandas
import os


MLNFL_ROOT_DIR = os.environ['MLNFL_ROOT']
PROBABILITY_FILE = MLNFL_ROOT_DIR + "/data/lookup/speead_to_probability.csv"


def load_probability(dataFile=PROBABILITY_FILE):

    df = pandas.read_csv(dataFile)
    return df



