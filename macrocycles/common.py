import pandas as pd
from enum import Enum

colormap = {
    "1": "#ff1f64",
    "2": "#009fe0",
    "3": "#ffc922",
    "4": "#b35ebe",
    "5": "#00d072",
    "linear peptide": "#a5a5a9",
    }

class BOND(Enum):
    PEPTIDE = 1
    ESTER = 2

    def __eq__(self, other):
        return self.value == other.value and self.name == other.name

    def __hash__(self):
        return self.value

class STEREO(Enum):
    MISSING = 1
    NOT_STEREOCENTER = 2
    NO_SUBSTITUENTS = 3
    MULTIPLE_SUBSTITUENTS = 4
    MULTIPLE_BACKBONE_CARBONS = 5
    FUSION = 6
    R = 7
    S = 8

    def __eq__(self, other):
        return self.value == other.value and self.name == other.name

class LD(Enum):
    L = 1
    D = 2
    OTHER = 3

    def __eq__(self, other):
        return self.value == other.value and self.name == other.name

class BACKBONE_TYPE(Enum):
    ALPHA = 1
    BETA = 2
    GAMMA = 3

    def __eq__(self, other):
        return self.value == other.value and self.name == other.name

class SIDE_CHAIN_STEREO:
    ALPHA_R = 1
    ALPHA_S = 2
    NON_ALPHA = 3
    OTHER = 4

    def __eq__(self, other):
        return self.value == other.value and self.name == other.name


class DIRECTION(Enum):
    PROXIMAL = 1
    DISTAL = 2

    def __eq__(self, other):
        return self.value == other.value and self.name == other.name

class SIDE_CHAIN_TYPE(Enum):

    CYCLIZATION = 1
    CANONICAL = 2
    NONCANONICAL = 3

    def __eq__(self, other):
        return self.value == other.value and self.name == other.name

class RESIDUE_TYPE(Enum):

    CYCLIZATION = 1
    CANONICAL = 2
    NONCANONICAL = 3

    def __eq__(self, other):
        return self.value == other.value and self.name == other.name

class ParsingData:

    molecule_df: pd.DataFrame
    match_df: pd.DataFrame

    def __init__(self, molecule_df, match_df):

        self.molecule_df = molecule_df
        self.match_df = match_df

    def loc(self, ids):

        molecule_df = self.molecule_df.copy().loc[ids]
        match_df = self.match_df.copy().loc[molecule_df["Molecule ID"].unique()]

        return ParsingData(molecule_df, match_df)

    def truncate(self, n=100):

        self.molecule_df = self.molecule_df.iloc[:n]
        self.match_df = self.match_df.loc[self.molecule_df["Molecule ID"].unique()]

    def get_structure_dict(self):

        d = {}

        for i in range(len(self.match_df)):

            ser = self.match_df.iloc[i]
            name = ser["Side chain name"]
            graph = ser["Everything graph"]

            d[name] = graph

        return d



