import pickle
import pandas as pd
import logging
from rdkit.Chem import PandasTools
from rdkit import Chem

#from macrocycles.parsing import Parser
from macrocycles import parsing
from macrocycles import utils
from macrocycles.common import ParsingData

class SDFIterator:

    def __init__(self, filename, id_fields, test_first = True):

        self.filename = filename
        self.id_fields = id_fields
        extension = filename.split(".")[-1].lower()

        if extension != "sdf":
            warnings.warn("File extension (.{extension}) is not sdf")

        if test_first:
            test_iterator = Chem.SDMolSupplier(self.filename)
            first_mol = test_iterator.__next__()
            self.get_id(first_mol)

        #test length
        f = open(self.filename, 'r')
        counter = 0
        for line in f:
            if "$$$$" in line:
                counter += 1

        f.close()
        self.length = counter

        self.iterator = Chem.SDMolSupplier(self.filename)

    def get_id(self, mol):

        if type(self.id_fields) == str:
            id_value = mol.GetProp(self.id_fields)
            return id_value
        elif type(self.id_fields) == list:
            values = []
            for prop in self.id_fields:
                value = mol.GetProp(prop)
                values.append(value)

            return tuple(values)
        else:
            raise Exception("'id_fields' must be string or list of strings, not {type(self.id_fields)}")

    def __len__(self):
        return self.length


    def __iter__(self):
        return self
        
    def __next__(self):

        mol = self.iterator.__next__()
        id_value = self.get_id(mol)

        return id_value, mol

class CSVIterator:

    def __init__(self, filename, mol_col, id_cols, delimiter = ",", test_first = True, ignore_header = False):

        self.filename = filename
        self.mol_col = mol_col
        self.id_cols = id_cols
        self.delimiter = delimiter
        extension = filename.split(".")[-1].lower()

        if extension != "csv":
            warnings.warn("File extension (.{extension}) is not sdf")

        if test_first:
            f = open(filename, 'r')
            if ignore_header:
                f.readline()
            first_line = f.readline()
            s = first_line.split(self.delimiter)
            #mol = Chem.MolFromSmiles(s[self.mol_col].strip())
            print(self.mol_col)
            mol = Chem.MolFromInchi(s[self.mol_col].strip())
            self.get_id(first_line) 
            f.close()

        self.iterator = open(filename, 'r')
        if ignore_header:
            self.iterator.readline()

        
        f = open(filename, 'r')
        counter = 0
        for line in f:
            counter += 1
        self.length = counter
        f.close()

    def get_id(self, line):

        if type(self.id_cols) == int:
            id_value = line.split(self.delimiter)[self.id_cols].strip()
            return id_value
        elif type(self.id_fields) == list:
            values = []
            for col in self.id_cols:
                value = line.split(self.delimiter)[col].strip()
                values.append(value)

            return tuple(values)
        else:
            raise Exception("'id_cols' must be int or list of ints, not {type(self.id_fields)}")

    def __len__(self):
        return self.length

    def __iter__(self):
        return self
        
    def __next__(self):

        line = self.iterator.readline()
        if len(line.strip()) == 0:
            self.iterator.close()
            raise StopIteration
        inchi = line.split(self.delimiter)[self.mol_col].replace('"', '').strip()

        mol = Chem.MolFromInchi(inchi)

        id_value = self.get_id(line)

        return id_value, mol

def get_mol_with_id(id_val, filename):

    cache_filename = ".cache/cached_mols.smi"

    try:
        f = open(cache_filename, 'r')

        #try cache
        for line in f:
            s = line.split(",")
            line_id_val = s[0].strip()
            smiles = s[1].strip()

            if id_val == line_id_val:
                f.close()
                return Chem.MolFromSmiles(smiles)
    except:
        try:
            f.close()
        except:
            pass

    try:
        f.close()
    except:
        pass
    mol = -1

    extension = filename.split("/")[-1].split(".")[-1].lower()
    if extension == "csv":

        f = open(filename, 'r')

        for line in f:
            s = line.split(",")
            line_id_val = s[0].strip()
            smiles = s[1].strip()

            if id_val == line_id_val:
                f.close()
                mol = Chem.MolFromSmiles(smiles)
                break

        f.close()
    elif extension == "sdf":


        f = open(filename, 'r')
        block = []
        for line in f:
            if "$$$$" in line:
                block = "".join(block)
                if id_val in block:
                    mol = Chem.MolFromMolBlock(block)
                    f.close()
                    break
                else:
                    block = []

            else:
                block.append(line)
        f.close()
    else:
        raise Exception(f"file extension {extension} not recognized")

    if mol == None:
        raise Exception("Mol not found")
    if mol == -1:
        raise Exception(f"Error reading mol: {id_val}")

    f.close()
    f = open(cache_filename, 'a+')
    f.write(f"{id_val},{Chem.MolToSmiles(mol)}\n")
    f.close()

    return mol


def get_saved_data(dirname):

    molecule_df = pd.read_pickle(f"{dirname}/molecules.pkl")
    match_df = pd.read_pickle(f"{dirname}/matches.pkl")

    few_match_ids = set()

    for molecule_id, count in dict(match_df["Molecule ID"].value_counts()).items():

        if count <= 3:
            few_match_ids.add(molecule_id)

    to_drop = []
    for i, row in molecule_df.iterrows():
        molecule_id = row["Molecule ID"]
        if molecule_id in few_match_ids:
            to_drop.append(molecule_id)
    print(f"{len(to_drop)} molecules have no residue matches, dropping...")

    molecule_df = molecule_df.drop(index=to_drop, errors="ignore")
    match_df = match_df.drop(index=to_drop, errors="ignore", level=0)

    d = {}
    for i in range(len(molecule_df)):
        ser = molecule_df.iloc[i]

        mol = ser["ROMol"]
        id_val = ser["Molecule ID"]
        d[id_val] = mol

    logging.info("Copying molecules to match df")
    mols = match_df["Molecule ID"].apply(lambda x: d[x])
    match_df["ROMol"] = mols

    return ParsingData(molecule_df, match_df)

def string_to_list(s):

    if pd.isnull(s):
        return []
    #strip off brackets
    s = s.strip()
    s = s[1:-1]
    #if "," not in s:
    #    return [int(s)]
    s = s.split(",")
    return_list = []
    for x in s:
        try:
            val = int(x)
            return_list.append(val)
        except:
            x = x[1:-1]
            if x[0] == "'":
                x = x[1:]
            return_list.append(x)

    return return_list



def get_mibigs_building_blocks():

    df = pd.read_csv("data/mibigs3_building_blocks.csv")
    return df
