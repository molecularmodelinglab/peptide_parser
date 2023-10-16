from macrocycles.setup import ROOT_DIR
from tqdm import tqdm
from PIL import Image
import io
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import os
import networkx as nx
import numpy as np
from itertools import combinations
from rdkit import Chem
from rdkit.Chem import SaltRemover
from networkx.algorithms import isomorphism
from multiprocessing import Pool
import copy
import shutil

import logging

from rdkit.Chem import Crippen


from macrocycles import graphs
from macrocycles import utils
from macrocycles.common import ParsingData
from macrocycles.common import BOND
from macrocycles.common import STEREO
from macrocycles.common import SIDE_CHAIN_TYPE
from macrocycles.common import RESIDUE_TYPE

from enum import Enum


CANONICAL_NAMES = set([
                "L-Alanine",
                "L-Arginine",
                "L-Asparagine",
                "L-Aspartate/Aspartic Acid",
                "L-Cysteine",
                "L-Glutamate/Glutamic Acid",
                "L-Glutamine",
                "L-Isoleucine",
                "L-Leucine",
                "L-Lysine",
                "L-Methionine",
                "L-Phenylalanine",
                "L-Serine",
                "L-Threonine",
                "L-Tryptophan",
                "L-Tyrosine",
                "L-Valine",
                "L-Proline",
                "L-Histidine",
                "Glycine"])


def node_match(node1, node2):

    if node1["element"] == "R" or node2["element"] == "R":
        return True

    if (node1["element"] == "CX" and
       len(node2["element"]) == 2 and
       node2["element"][0] == "C"):
        return True

    if (node2["element"] == "CX" and
       len(node1["element"]) == 2 and
       node1["element"][0] == "C"):
        return True

    if node1["element"] == "N" and node2["element"] == "N|PX":
            return True

    if node1["element"] == "N|PX" and node2["element"] == "N":
            return True

    else:

        if type(node1["element"]) != list:
            element1 = set([node1["element"]])
        else:
            element1 = set(node1["element"])
        if type(node2["element"]) != list:
            element2 = set([node2["element"]])
        else:
            element2 = set(node2["element"])
    if len(element1.intersection(element2)) > 0:
        return True
    else:
        return False

def get_residue_name(ser):

    side_chain_name = ser["Side chain name"]

    s = ""
    if ser["cat N-methyl"]:
        s += "[Nmet]"
    if ser["cat Non-methyl nitrogen modification"]:
        s += "[Non-met Nmod]"
    if ser["Proximal bond"] == BOND.ESTER:
        s += "[O]"

    return s + side_chain_name
     

def get_name(ser):

    side_chain_name = ser["Side chain name"]
    side_chain_name = shorten_name(side_chain_name)

    if side_chain_name != "Glycine":
        stereo = ser["Side chain carbon stereo"]
        if side_chain_name in canonical_names and stereo in [STEREO.R, STEREO.S]:
 
            if stereo == STEREO.S and side_chain_name == "Cysteine":
                stereo = "D-"
            elif stereo == STEREO.R and side_chain_name != "Cysteine":
                stereo = "D-"
            else:
                stereo = "L-"
 
        elif stereo not in [STEREO.R, STEREO.S]:
            #stereo = "[Complicated stereo]"
            if stereo == STEREO.MULTIPLE_BACKBONE_CARBONS:
                stereo = ""
            elif stereo == STEREO.NOT_STEREOCENTER:
                stereo = ""
            else:
                stereo = f"[{stereo}]"
        else:
            stereo = "[" + str(stereo).split(".")[-1] + "]"
    else:
        stereo = ""

    s = ""
    if ser["cat N-methyl"]:
        s += "[Nmet]"
    if ser["cat Non-methyl nitrogen modification"]:
        s += "[Non-met Nmod]"
    if ser["Proximal bond"] == BOND.ESTER:
        s += "[O]"
    if ser["Backbone type"] == "B":
        s += "[Beta]"
        stereo = ""
    elif ser["Backbone type"] == "G":
        s += "[Gamma]"
        stereo = ""
    s += f"{stereo}"

    s += side_chain_name

    return s


def parallelize_dataframe(df, function, n_cores=32):
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    data = pool.map(function, df_split)
    new_df = pd.concat(data)
    pool.close()
    pool.join()
    return new_df

def get_tier(ser, matches, max_side_chain_length = 15, thiopeptide_allowed = True):

    if "Reason for exclusion" in ser and pd.notnull(ser["Reason for exclusion"]):
        reason = ser["Reason for exclusion"].replace(" ","_")
        return f"excluded/{reason}"

    if matches is None:
        return "excluded/non_peptide"

    #a cyclization (internal or terminal) must reach this many residues away to be considered a cycle
    cyclization_distance_threshold = 2

    actual_matches = matches[(~matches["Internal cyclization"] & ~matches["Terminal cyclization"])]

    feasible_matches = []

    for i, match in matches.iterrows():
        if match["Internal cyclization"]:
            if match["Internal cyclization distance"] > cyclization_distance_threshold:
                feasible_matches.append(match)
        elif match["Terminal cyclization"]:
            if match["Terminal cyclization distance"] > cyclization_distance_threshold:
                feasible_matches.append(match)
        elif match["Side chain name"] == "Multiple_events":
            feasible_matches.append(match)
        else:
            feasible_matches.append(match)

    side_chain_lengths = actual_matches["Side chain atom indices"].apply(len)
    big_side_chains = [x > max_side_chain_length for x in side_chain_lengths]

    internal_cyclizations = (matches[(matches["Internal cyclization"]) &
                            (matches["Internal cyclization distance"] > cyclization_distance_threshold)])

    terminal_cyclizations = (matches[(matches["Terminal cyclization"]) &
                            (matches["Terminal cyclization distance"] > cyclization_distance_threshold)])

    all_cyclizations = \
        matches[matches["Internal cyclization"] | matches["Terminal cyclization"]]

    side_chain_side_chain = []
    side_chain_side_chain_side_chains = set()

    # check for identical cyclization graphs
    for i, cyclization in all_cyclizations.iterrows():
        for j, other_cyclization in all_cyclizations.iterrows():
            if i >= j:
                continue

            if set(cyclization["Side chain atom indices"]) == set(other_cyclization["Side chain atom indices"]):
                side_chain_side_chain.append((i, j))
                side_chain_side_chain_side_chains.add(i)
                side_chain_side_chain_side_chains.add(j)

    any_side_chain_side_chain = False
    if len(side_chain_side_chain) > 0:
        any_side_chain_side_chain = True

    # any_side_chain_backbone = (set(range(len(all_cyclizations))) != side_chain_side_chain_side_chains)
    any_side_chain_backbone = len(set(all_cyclizations.index) - set(side_chain_side_chain_side_chains)) > 0

    any_multiple_internal = any(matches["Multiple internal cyclizations"])
    any_multiple_terminal = any(matches["Multiple terminal cyclizations"])
    any_internal_and_terminal = any(matches["Internal and terminal cyclizations"])

    proximal_bond_types = set(matches["Proximal bond"])
    distal_bond_types = set(matches["Distal bond"])

    mol = ser["ROMol"]

    
    ri = mol.GetRingInfo()
    largest_ring_size = max((len(r) for r in ri.AtomRings()), default=0)
    has_large_ring = (largest_ring_size > 8)

    #print(proximal_bond_types)
    #rint(distal_bond_types)

    tier = None
    if len(feasible_matches) <= 3:
        return "excluded/too_few_matches"

    elif has_thiopeptide_pattern(mol):
        return "peptide/macrocycle/4/4A_detected_thiopeptide"

    elif BOND.PEPTIDE not in distal_bond_types or BOND.PEPTIDE not in proximal_bond_types:
        return "excluded/only_ester_matches"

    elif ser["Cyclic backbone"]:
        if len(terminal_cyclizations) > 0:
            raise Exception("Cyclic backbone with terminal cyclization?")

        if not any_side_chain_side_chain and not any_side_chain_backbone:

            #any_fatty_acid = any(matches["Side chain graph"].apply(is_fatty_acid))
            any_fatty_acid = any(matches["cat Fatty Acid"])
            if any_fatty_acid:
                return "peptide/macrocycle/1/1B_fatty_acid"
            else:
                return "peptide/macrocycle/1/1A"

        elif any_side_chain_backbone and not any_side_chain_side_chain:

            if len(internal_cyclizations) > 1:

                return "peptide/macrocycle/1/1F_multiple_internal_cyclization"

            else:
                return "peptide/macrocycle/5/5A_backbone_with_internal_cyclization"

        elif any_side_chain_side_chain and not any_side_chain_backbone:

            if len(side_chain_side_chain) > 1:
                return "peptide/macrocycle/5/5C_backbone_with_multiple_side_chain_side_chain"
            else:
                return "peptide/macrocycle/5/5B_backbone_with_side_chain_side_chain"

        elif any_side_chain_side_chain and any_side_chain_backbone:

            return "peptide/macrocycle/1/1E_multiple_other_cyclization"

        else:
            raise Exception("Should be unreachable 1")

    elif any_internal_and_terminal:
        return "peptide/macrocycle/5/5I_single_residue_both_ss_and_sb"
    elif any_multiple_internal or any_multiple_terminal:
        if any_multiple_internal and any_multiple_terminal:
            return "peptide/macrocycle/5/5F_multiple_multiple"
        elif any_multiple_internal:
            return "peptide/macrocycle/5/5G_multiple_same_side_chain_side_chain"
        elif any_multiple_terminal:
            return "peptide/macrocycle/5/5H_multiple_same_side_chain_backbone"


    elif not any_side_chain_side_chain and not any_side_chain_backbone:

        if has_large_ring:
            return "peptide/linear_but_large_ring/linear_peptide_but_large_ring"
        else:
            return "peptide/linear/linear_peptide"
    
    elif any_side_chain_backbone and not any_side_chain_side_chain:


        if len(terminal_cyclizations) > 0 and len(internal_cyclizations) == 0:

            if len(terminal_cyclizations) > 1:
                return "excluded/handle/mutiple_terminal_cyclizations"
            elif len(terminal_cyclizations) == 1:

                terminal_cyclization_atoms = terminal_cyclizations.iloc[0]["Side chain atom indices"]
                if len(terminal_cyclization_atoms) > 10:
                    return "peptide/macrocycle/2/2C_large_terminal_cyclization"
                elif sum(big_side_chains) > 0:
                    return "peptide/macrocycle/2/2D_large_side_chain"
                elif matches.iloc[-1]["Terminal cyclization"]:
                    return "peptide/macrocycle/2/2B_terminal_terminal_cyclization"
                elif len(terminal_cyclizations) == 1:
                    return "peptide/macrocycle/2/2A_lariat"

        elif len(internal_cyclizations) > 0 and len(terminal_cyclizations) == 0:
            return "peptide/macrocycle/2/2E_only_internal"

        elif len(terminal_cyclizations) > 0 and len(internal_cyclizations) > 0:
            return "peptide/macrocycle/5/5D_multiple_side_chain_to_backbone_types"

        else:
            return "peptide/uncategorized/uncategorized_peptide"

    # elif any_side_chain_side_chain:
    elif any_side_chain_side_chain and not any_side_chain_backbone:

        if len(side_chain_side_chain) > 1:
            return "peptide/macrocycle/5/5E_multiple_side_chain_side_chain"
        elif len(side_chain_side_chain) == 1:
            return "peptide/macrocycle/3/3A_side_chain_side_chain"
        else:
            raise Exception("Should be unreachable 3")

    elif any_side_chain_side_chain and any_side_chain_backbone:
        return "peptide/macrocycle/5/5_multiple_cyclization_types"

    if tier == None:
        return "unhandled"

    return tier

def get_simple_class(ser):

    tier = ser["Class"]

    s = tier.split("/")
    if s[0] == "peptide" and s[1] == "macrocycle":
        simple_class = tier.split("/")[-2]
    elif s[0] == "peptide":
        simple_class = "linear peptide"
    else:
        print(s)
        simple_class = "non_peptide"

    return simple_class

def remove_duplicates(molecule_data):

    molecule_df = molecule_data.molecule_df.copy()
    match_df = molecule_data.match_df.copy()

    duplicated = molecule_df[molecule_df.duplicated(subset = "Molecule ID")]
    if len(duplicated) > 0:
        logging.warn(f"Molecule DF contains duplicates, dropping them")
        molecule_df = molecule_df.drop_duplicates(subset = "Molecule ID")
        logging.info(f"After dropping duplicates: {len(molecule_df)}")

        molecule_ids = list(molecule_df["Molecule ID"].unique())
        molecule_ids = [x for x in molecule_ids if x in match_df["Molecule ID"]]
        #valid_match = match_df["Molecule ID"].apply(lambda x: x in molecule_ids)
        #match_df = match_df[valid_match]
        match_df = match_df.loc[molecule_ids]

    return ParsingData(molecule_df, match_df)

def has_pyridine(graph):


    def this_node_match(node1, node2):

        if node1["element"] == "R" or node2["element"] == "R":
            return True

        if (node1["element"] == "CX" and
           node2["element"] == "C"):
            return True

        if (node2["element"] == "CX" and
           node1["element"] == "C"):
            return True

        if node1["element"] == "N" and node2["element"] == "N|PX":
                return True

        if node1["element"] == "N|PX" and node2["element"] == "N":
                return True

        else:

            if type(node1["element"]) != list:
                element1 = set([node1["element"]])
            else:
                element1 = set(node1["element"])
            if type(node2["element"]) != list:
                element2 = set([node2["element"]])
            else:
                element2 = set(node2["element"])
        if len(element1.intersection(element2)) > 0:
            return True
        else:
            return False


    pyridine_graph = nx.Graph()
    pyridine_graph.add_nodes_from([(1, {"element":"C"}),
                                   (2, {"element":"C"}),
                                   (3, {"element":"C"}),
                                   (4, {"element":"C"}),
                                   (5, {"element":"C"}), 
                                   (6, {"element":"N"}),
                                   ])

    pyridine_graph.add_edges_from([(1,2),
                                   (2,3),
                                   (3,4),
                                   (4,5),
                                   (5,6),
                                   (6,1)])

    GM = isomorphism.GraphMatcher(graph, pyridine_graph, node_match = this_node_match)
    return GM.subgraph_is_isomorphic()




def has_thiopeptide_pattern(mol):


    mol_graph, _ = graphs.make_graph(mol, compute_hash = False)

    pyridine_graph = nx.Graph()
    pyridine_graph.add_nodes_from([(1, {"element":"C"}),
                                   (2, {"element":"C"}),
                                   (3, {"element":"C"}),
                                   (4, {"element":"C"}),
                                   (5, {"element":"C"}), 
                                   (6, {"element":"N"}),
                                   (7, {"element":"C"}),
                                   (8, {"element":"N"}),
                                   (9, {"element":"C"}),
                                   (10, {"element":"N"}),
                                   (11, {"element":["O", "S"]}),
                                   (12, {"element":"C"}),
                                   (13, {"element":"N"}),
                                   (14, {"element":["O","S"]}),
                                   ])

    pyridine_graph.add_edges_from([(1,2),
                                   (2,3),
                                   (3,4),
                                   (4,5),
                                   (5,6),
                                   (6,1),
                                   (5,7),
                                   (7,8),
                                   (4,9),
                                   (9,10),
                                   (9,11),
                                   (1,12),
                                   (12,13),
                                   (12,14),
                                   ])
 
    GM = isomorphism.GraphMatcher(mol_graph, pyridine_graph, node_match = node_match)
    return GM.subgraph_is_isomorphic()

def get_molecule_volume(df):

    from rdkit.Chem import rdDistGeom
    from rdkit.Chem import AllChem

    embed_params = rdDistGeom.ETKDGv3()

    from tqdm import tqdm

    volumes = []
    for i in tqdm(range(len(df))):

        s = df.iloc[i]
        inchi = s["Inchi"]
        try:
            mol = Chem.MolFromInchi(inchi)
            AllChem.EmbedMolecule(mol, embed_params)
            volume = AllChem.ComputeMolVolume(mol)
        except:
            volume = None

        volumes.append(volume)

    volume_df = pd.DataFrame(volumes, columns = ["ETKDGv3 volume"])
    return volume_df

def add_molecule_volume(df):
    df = df.copy(deep = True)
    volume_df = parallelize_dataframe(df, get_molecule_volume, n_cores = 32)
    volume_df = volume_df.set_index(df.index)
    df = pd.concat((df, volume_df), axis = 1)
    return df

def get_rdkit_descriptors(mol):

    desc_dict = {}
    molwt = Chem.rdMolDescriptors.CalcExactMolWt(mol)
    desc_dict["Molecular weight"] = molwt

    hba = Chem.rdMolDescriptors.CalcNumHBA(mol)
    desc_dict["Num hydrogen bond acceptors"] = hba

    hbd = Chem.rdMolDescriptors.CalcNumHBD(mol)
    desc_dict["Num hydrogen bond donors"] = hbd

    rot_bond = Chem.rdMolDescriptors.CalcNumRotatableBonds(mol)
    desc_dict["Num rotatable bonds"] = rot_bond

    tpsa = Chem.rdMolDescriptors.CalcTPSA(mol)
    desc_dict["Topological Polar Surface Area"] = tpsa

    logp = Chem.Crippen.MolLogP(mol)
    desc_dict["LogP"] = logp

    csp3 = Chem.rdMolDescriptors.CalcFractionCSP3(mol)
    desc_dict["CSP3"] = csp3

    return desc_dict

def merge_datasets(curate = False):

    log_filename = f"merge_log.txt"
    logging.basicConfig(filename=log_filename, level=logging.DEBUG, filemode = 'w')

    filenames = [("Supernatural 2", "data/SupernaturalII_Data_Charite.csv"), ("Lotus", "data/lotus.csv"), ("Mibigs", "data/mibigs_processed.csv")]

    def curate_mol(mol):

        old_mol = copy.copy(mol)

        #removal of mixtures
        fragmenter_object = molvs.fragment.LargestFragmentChooser(prefer_organic = True)
        newmol = fragmenter_object.choose(mol)
        if newmol is None:
            return None

        #removal of inorganics
        if not molvs.fragment.is_organic(mol):
            return None

        #removal of salts
        remover = SaltRemover.SaltRemover()
        newmol = remover.StripMol(mol, dontRemoveEverything=True) #tartrate is listed as a salt? what do?
        if newmol is None:
            return None

        #structure normalization
        normalizer = molvs.normalize.Normalizer(normalizations=molvs.normalize.NORMALIZATIONS,
                max_restarts = molvs.normalize.MAX_RESTARTS)
        newmol = normalizer.normalize(mol)
        if newmol is None:
            return None

        #tautomer selection
        tautomerizer = molvs.tautomer.TautomerCanonicalizer(transforms=molvs.tautomer.TAUTOMER_TRANSFORMS, scores =
                molvs.tautomer.TAUTOMER_SCORES, max_tautomers=molvs.tautomer.MAX_TAUTOMERS)
        newmol = tautomerizer(mol)
        if newmol is None:
            return None
        
        #disconnect metals
        metal_remover = molvs.metal.MetalDisconnector()
        newmol = metal_remover.disconnect(mol)
        if newmol is None:
            return None

        return mol

    from joblib import Parallel, delayed

    d = {}

    ids = []
    dataset_names = []
    mols = []
    print("Reading molecules")
    for dataset_name, filename in filenames:

        f = open(filename, 'r')
        for i, line in enumerate(f):

            s = line.split(",")
            id_val = s[0]
            smiles = s[1]
            mol = Chem.MolFromSmiles(smiles)
            if mol == None:
                logging.info(f"FAILED TO READ: {dataset_name}, {id_val}")
                continue
            mol.SetProp("Identifier", id_val)
            mol.SetProp("Source dataset", dataset_name)
            ids.append(id_val)
            dataset_names.append(dataset_name)
            mols.append(mol)

    if curate:
        print("Starting parallel curation")
        #batch_size of 1 to ensure that a long-running molecule will not hold up any others assigned to that process
        mols = Parallel(n_jobs = 30, batch_size = 1)(delayed(curate_mol)(mol) for mol in tqdm(mols))
        print("Finished parallel curation")


    #add test data to ensure logging is working correctly
    mols.append(None)
    ids.append("fake id")
    dataset_names.append("fake dataset")

    print("Deduplicating and writing output")
    for i in range(len(mols)):
        mol = mols[i]
        id_val = ids[i]
        dataset_name = dataset_names[i]
        if mol is None:
            logging.info(f"FAILED TO CURATE MOL: {dataset_name}, {id_val}")
            continue
        mol.SetProp("Identifier", id_val)
        mol.SetProp("Source dataset", dataset_name)
        inchi = Chem.MolToInchi(mol)
        if inchi not in d:
            d[inchi] = []
        d[inchi].append(mol)

    writer = Chem.SDWriter("merged_datasets.sdf")

    #deduplicate, but keep track of source datasets/ids
    for key, value in d.items():

        if len(value) > 1:
            datasets = set([x.GetProp("Source dataset") for x in value])
            identifiers = set([x.GetProp("Identifier") for x in value])
            mol = value[0]
            mol.SetProp("Source dataset", ",".join(datasets))
            mol.SetProp("Identifier", ",".join(identifiers))
            logging.info(f"DEDUPLICATING: {mol.GetProp('Source dataset')}, {mol.GetProp('Identifier')}, {len(value)} original entries\n")
        else:
            mol = value[0]

        writer.write(mol)

def shorten_name(name):

    if name[0] == "#":
        return name[:9]
    else:
        return name

def draw_accessibility_in_folder(molecule_df, match_df, output_path):
 
    try:
        shutil.rmtree(output_path)
    except Exception as e:
        print(e)
        pass

    try:
        os.makedirs(output_path)
    except Exception as e:
        print(e)
    
    data_for_parallel = []

    for i in range(len(molecule_df)):

        mol_ser = molecule_df.iloc[i]
        sub_match_df = match_df.loc[mol_ser["Molecule ID"]]

        inaccessible_count = sum(sub_match_df["Residue accessible"] == False)

        dirname = mol_ser["Class"] + f"/{inaccessible_count}_inaccessible/"

        stem = output_path + "/" + dirname

        try:
            os.makedirs(stem)
        except Exception as e:
            pass
        

        matches = match_df.loc[mol_ser["Molecule ID"]]
        output_filename = stem + mol_ser["Molecule ID"] + ".svg"
        data_for_parallel.append((mol_ser, sub_match_df, output_filename))

    from multiprocessing import Pool
    with Pool(25) as p:

        p.map(draw_accessibility_for_parallel, data_for_parallel)

def draw_accessibility_for_parallel(task):

    from macrocycles import drawing
    mol_ser, sub_match_df, output_filename = task

    svg_text = drawing.highlight_accessibility(mol_ser, sub_match_df)


    f = open(output_filename, 'wb')
    f.write(svg_text)
    f.close()

def copy_to_folder(source_dirname, molecule_df, output_path):
    
    try:
        shutil.rmtree(output_path)
    except:
        pass
    try:
        os.makedirs(output_path)
    except Exception as e:
        print(e)
    
    for i in range(len(molecule_df)):

        mol_ser = molecule_df.iloc[i]
        start_filename =f"{source_dirname}/images/{mol_ser['Molecule ID']}.html"
        end_filename = output_path + "/" + mol_ser["Molecule ID"] + ".html"
        shutil.copyfile(start_filename, end_filename)

def get_image_path(mol_ser):

    tier = mol_ser["Class"]

    return tier + "/" + mol_ser["Molecule ID"]

def draw_for_parallel(graph, draw_func):
    print(graph, draw_func)
    
    if graph is None:
        return None

    try:
        return draw_func(graph)
    except Exception as e:
        print("HERE EXCEPT")
        print(e)
        return None


def parallel_graph_draw(graph_list, draw_func):

    from multiprocessing import Pool
    from functools import partial

    with Pool(20) as p:

        results = p.map(partial(draw_for_parallel, draw_func = draw_func), graph_list)

    return results


def df_plus_side_chains(df, output_filename, match_df, side_chain_col = "Side chain name", graph_name = "Everything graph", draw_func = graphs.draw_side_chain):

    if side_chain_col not in df.columns and side_chain_col != "index":
        raise Exception()

    if side_chain_col == "index":
        side_chains = df.index
    else:
        side_chains = df[side_chain_col]


    structure_dict = {}
    

    for i in range(len(match_df)):
        ser = match_df.iloc[i]
        name = ser[side_chain_col]
        graph = ser[graph_name]
        structure_dict[name] = graph

    graph_list = []
    for x in side_chains:
        if x in structure_dict:
            try:
                graph_list.append(structure_dict[x])
            except:
                graph_list.append(None)
        else:
            graph_list.append(None)


    image_list = parallel_graph_draw(graph_list, draw_func)

    graph_list = image_list

    import xlsxwriter
    import os

    output_dirname = "/".join(output_filename.split("/")[:-1])
    try:
        os.makedirs(output_dirname, exist_ok = True)
    except:
        pass

    print(output_filename)
    writer = pd.ExcelWriter(output_filename, engine = 'xlsxwriter')
    df.to_excel(writer, sheet_name = "Sheet1", index = True)
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']
    print("B")

    for i in range(1, len(df) + 1):
        #worksheet.set_row(i, 400)
        worksheet.set_row(i, 120)

    text_format = workbook.add_format({'font_size':20, 'align':'center', 'valign':'center'})
    worksheet.set_column("B:G", 50, text_format)
    text_format = workbook.add_format({'font_size':20, 'align':'center', 'valign':'center'})
    worksheet.set_column("A:A", 50, text_format)

    #thank you stackoverflow
    def buffer_image(image: Image, format: str = 'PNG'):
        if image == None:
            return None, None
        buf = io.BytesIO()
        image.save(buf, format=format)
        return buf, image


    target_height = 160

    print("C")
    for i, graph in enumerate(graph_list):
        buf, image = buffer_image(graph)
        print(buf) 
        if buf != None:
            #worksheet.insert_image(i + 1, 3, "fake label", {'image_data': buf, 'object_position':3, 'x_scale':0.5, 'y_scale': 0.5})
            #worksheet.insert_image(i + 1, len(df.columns) + 1, "fake label", {'image_data': buf, 'object_position':3, 'x_scale':0.5, 'y_scale': 0.5})
            print(image.width, image.height)

            #only care about height
            scale = target_height / image.height

            worksheet.insert_image(i + 1, len(df.columns) + 1, "fake label", {'image_data': buf, 'object_position':2, 'x_scale':scale, 'y_scale': scale})
    writer.save()

def deduplicate_stereo(molecule_data, input_dirname, output_dirname):

    def get_flat_inchi(mol):

        import copy
        new_mol = copy.copy(mol)

        from rdkit.Chem import rdmolops
        rdmolops.RemoveStereochemistry(new_mol)

        inchi = Chem.MolToInchi(new_mol)
        return inchi

    def align_cycle_sequences(linear_sequence, cycle_sequence):
    
        for i in range(len(cycle_sequence)):
            this_seq = []
            remap = []
            
            for j in range(len(cycle_sequence)):
                idx = (i + j) % len(cycle_sequence)
                this_seq.append(cycle_sequence[idx])
                remap.append(idx)
            if this_seq == linear_sequence: 
                return remap
            
        return list(range(len(linear_sequence)))
        #raise Exception("Failed to align")

    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')

    molecule_df = molecule_data.molecule_df.copy()

    os.makedirs(output_dirname, exist_ok = True)

    # # # # # # # # # # # # # # # # 
    # generate flat inchis for all molecules
    # # # # # # # # # # # # # # # # 
    if "Flat Inchi" not in molecule_df.columns:
        molecule_df["Flat Inchi"] = molecule_df["ROMol"].apply(get_flat_inchi)


    cluster_counts = dict(molecule_df["Flat Inchi"].value_counts())

    cluster_inchis = set()
    for inchi, count in cluster_counts.items():
        if int(count) > 1:
            cluster_inchis.add(inchi)

    all_to_delete = {}
    universal_to_remove = set()
    for idx, cluster_inchi in tqdm(enumerate(cluster_inchis), total = len(cluster_inchis)):

        mol_ids = molecule_df[molecule_df["Flat Inchi"] == cluster_inchi]["Molecule ID"]

        anchor_matches = molecule_data.match_df.loc[mol_ids[0]]
        matches_to_transform = [molecule_data.match_df.loc[x] for x in mol_ids[1:]]

        transformed_set = list()
        
        keep_cluster = True
        for transform_match in matches_to_transform:
            
            transformation = align_cycle_sequences(list(anchor_matches["Side chain name"]), list(transform_match["Side chain name"]))
            
            try:
                transformed = transform_match.iloc[transformation]
            except:
                logging.warning("SOME DEDUPLICATIONS SETS ARE DISAGREEING, REMOVING ALL FOR NOW")
                keep_cluster = False

            transformed_set.append(transformed)
            
        if keep_cluster:
            transformed_set.append(anchor_matches)
        else:
            [universal_to_remove.add(x) for x in mol_ids]
            continue
        
        sequences = [list(x["Side chain name"]) for x in transformed_set]
        stereo = [list(x["Side chain carbon stereo"]) for x in transformed_set]
        names = [x.iloc[0]["Molecule ID"] for x in transformed_set]
        
        sers = []
        stereo_sers = []
        for i in range(len(names)):
            ser = pd.Series(sequences[i], name = names[i])
            stereo_ser = pd.Series(stereo[i], name = names[i])
            stereo_sers.append(stereo_ser)
            sers.append(ser)
            
        
        stereo_sers = [pd.Series(list(anchor_matches["Side chain name"]), name = "Shared side chain name")] + stereo_sers
        stereo_df = pd.DataFrame(stereo_sers).transpose()
        #display(stereo_df)
        
        this_dir = output_dirname + f"/cluster_{idx}"

        copy_to_folder(input_dirname, molecule_df.loc[mol_ids], this_dir)


        def run_dedup_options(molecule_df):

            def fully_covered(molecule_data):

                ids = molecule_data.molecule_df.columns[1:]
                from itertools import combinations
                combs = list(combinations(ids, 2))

                to_delete = set()
                for comb in combs:

                    stereo_a = np.array(molecule_data.molecule_df[comb[0]])
                    stereo_b = np.array(molecule_data.molecule_df[comb[1]])

                    id_a = comb[0]
                    id_b = comb[1]

                    nones = (stereo_a == STEREO.MISSING) | (stereo_b == STEREO.MISSING)
                    if len(nones) == 0:
                        continue

                    both = np.vstack((stereo_a, stereo_b)).transpose()

                    to_inspect = both[nones]

                    a_covers = [x[0] != STEREO.MISSING
                                and x[1] == STEREO.MISSING for x in to_inspect]

                    b_covers = [x[1] != STEREO.MISSING
                                and x[0] == STEREO.MISSING for x in to_inspect]

                    a_totally_covers = sum(a_covers) > 0 and sum(b_covers) == 0
                    b_totally_covers = sum(b_covers) > 0 and sum(a_covers) == 0

                    if a_totally_covers and b_totally_covers:
                        raise Exception("Why though")

                    if a_totally_covers:
                        to_delete.add(id_b)
                    if b_totally_covers:
                        to_delete.add(id_a)

                    continue

                #delete_vec = [x in to_delete for x in stereo_df.columns]

                #stereo_df.loc["Delete"] = delete_vec
                #stereo_df.to_csv(this_dir + f"/cluster_{idx}_stereo_info.csv", na_rep = "None", index = True)

                return to_delete


            ##DOES NOT CHECK IF THE COVERING MOLECULE HAS THE SAME DEFINED STEREO
            # a     b
            # ---------
            # R     S
            # R     None
            #
            # will remove b
            def combined_any_covered(molecule_data):

                from itertools import combinations
                combs = list(combinations(molecule_df, 2))

                to_delete = set()

                for column in molecule_df.columns[1:]:

                    other_stereo_coverage = molecule_df[[x for x in molecule_df.columns[1:] if x != column]]
                    other_stereo_coverage = (other_stereo_coverage != STEREO.MISSING).to_numpy()
                    others_have_stereo_info = (np.sum(other_stereo_coverage, axis = 1) > 0).astype(int)

                    this_stereo_coverage = (molecule_df[column] != STEREO.MISSING).to_numpy().astype(int)


                    covered = others_have_stereo_info - this_stereo_coverage
                    if any(covered > 0) and all(covered >= 0):
                        to_delete.add(column)
                            

                return to_delete

            def any_covered(molecule_df):
                raise NotImplementedError
            strategies = {"fully_covered": fully_covered,
                          "combined_any_covered": combined_any_covered,
                          }

            results = {}
            for strategy, dedup_func in strategies.items():


                try:
                    deduped = dedup_func(molecule_data)
                    results[strategy] = deduped
                except Exception as e:
                    print(f"{strategy}: {e}")

            return results

        results = run_dedup_options(stereo_df)

        for k, v in results.items():
            [v.add(x) for x in universal_to_remove]

        def update_dict(main, new):

            for key, value in new.items():
                if key not in main:
                    main[key] = value
                else:
                    main[key] = main[key].union(value)

            return main

        all_to_delete = update_dict(all_to_delete, results)

    def fully_specified(molecule_df, match_df):

        from rdkit.Chem.rdchem import StereoSpecified

        from rdkit.Chem.rdmolops import FindPotentialStereo


        to_delete = set()

        for id_val in match_df["Molecule ID"].unique():

            mol = molecule_df.loc[id_val]["ROMol"]

            stereoinfo = FindPotentialStereo(mol)

            missing = [x.specified == StereoSpecified.Unspecified for x in stereoinfo]

            if any(missing):
                to_delete.add(id_val)

        return to_delete

    def any_specified(molecule_df, match_df):

        to_delete = set()

        for column in molecule_df.columns[1:]:

            if all(molecule_df[column] == STEREO.MISSING):
                to_delete.add(column)

        return to_delete

    def side_chain_specified(molecule_df, match_df):

        to_delete = set()

        for id_val in match_df["Molecule ID"].unique():
            stereo = match_df.loc[id_val]["Side chain carbon stereo"]

            if any(stereo == STEREO.MISSING):
                to_delete.add(id_val)

        return to_delete




    def nothing(molecule_df):
        return set()



    '''
    whole_strategies = {
                  "fully_specified": fully_specified,
                  "any_specified": any_specified,
                  "do_nothing": nothing,
                  }
    '''


    whole_strategies = {
                  "fully_specified": fully_specified,
                  "side_chain_specified": side_chain_specified,
                  }


    for strategy, function in whole_strategies.items():
        to_delete = function(molecule_df, molecule_data.match_df)
        all_to_delete[strategy] = to_delete





    output = {}
    for strategy, to_delete in all_to_delete.items():

        this_molecule_df = molecule_df.loc[molecule_df.index[~molecule_df.index.isin(to_delete)]]
        this_match_df = molecule_data.match_df.loc[this_molecule_df.index]

        removed_molecule_df = molecule_df.loc[to_delete]
        removed_match_df = molecule_data.match_df.loc[to_delete]

        output[strategy] = {}
        output[strategy]["kept"] = (this_molecule_df.copy(), this_match_df.copy())
        output[strategy]["removed"] = (removed_molecule_df.copy(), removed_match_df.copy())



    return output

def longest_run(values, cyclic = False):

    if all(values):
        return len(values)

    if cyclic:
        runs = []
        for start in range(len(values)):
            run = 0
            for i in range(len(values)):
                idx = (start + i) % len(values)
                if values[idx]:
                    run += 1
                else:
                    if run > 0:
                        runs.append(run)
                    run = 0
            if run > 0:
                runs.append(run)

        if len(runs) == 0:
            longest_run = 0
        else:
            longest_run = max(runs)


    else:
        runs = []
        run = 0
        for i in range(len(values)):
            if values[i]:
                run += 1
            else:
                if run > 0:
                    runs.append(run)
                run = 0

        if run > 0:
            runs.append(run)

        if len(runs) == 0:
            longest_run = 0
        else:
            longest_run = max(runs)
            
    return longest_run

def seq_equals(a, b):

    if len(a) != len(b):
        return False

    for i in range(len(a)):
        built = []
        idx = i
        while len(built) < len(a):
            built.append(a[idx % len(a)])
            idx += 1
            
        if built == b:
            return True

    return False





if __name__ == "__main__":
    draw_example()
