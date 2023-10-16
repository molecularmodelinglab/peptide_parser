from typing import List
import traceback
import random
import string
import copy
import time
import numpy as np
import pandas as pd
from rdkit import Chem
import networkx as nx
from networkx.algorithms import isomorphism
import logging
from rdkit.Chem import rdCIPLabeler
from rdkit.Chem.rdchem import RWMol
from enum import Enum
from dataclasses import dataclass

from . import utils
from .common import STEREO
from .common import LD
from .common import BACKBONE_TYPE
from .common import BOND
from .common import SIDE_CHAIN_TYPE
from .common import RESIDUE_TYPE
from .common import DIRECTION

from . import graphs
from . import accessibility
from .setup import ROOT_DIR

# used to signal improper execution
class ParsingException(Exception):
    pass


# used to signal too much parsing time
class TimeException(Exception):
    pass

class Event:

    def __init__(self, type, idx, attributes = {}):

        self.type = type
        self.idx = idx
        self.attributes = attributes

    def __repr__(self):
        return f"{self.type}: {self.idx}, {self.attributes}"


class EVENT_TYPE(Enum):

    TERMINAL_CYCLIZATION = 3
    INTERNAL_CYCLIZATION = 4
    SELF_LOOP = 6


@dataclass
class SideChain():

    kind: EVENT_TYPE
    graph: nx.graph.Graph
    name: str
    distance: int

@dataclass
class SideChainInfo():
    
    events: list
    graph: nx.graph.Graph
    full_backbone_graph: nx.graph.Graph
    with_stereo_graph: nx.graph.Graph
    side_chain_hash: str
    orderless_side_chain_hash: str
    orderless_graph: nx.graph.Graph
    full_backbone_side_chain_hash: str
    with_stereo_hash: str
    legacy_graph: nx.graph.Graph
    legacy_hash: str
    everything_graph: nx.graph.Graph
    everything_hash: str

def get_named_hashes():

    df = pd.read_csv("data/side_chain_structures.txt", index_col = None, sep =",")
    d = {}

    p1 = df[df["Priority"] == 1]
    p2 = df[df["Priority"] == 2]
    canonical = df[df["Canonical"] == True]

    p1_d = {}
    for i in range(len(p1)):
        ser = p1.iloc[i]
        succeeded, info = get_info(ser["SMILES"])
        assert(succeeded is True)
        name = ser["Name"]

        p1_d[info.everything_hash] = name

    p2_d = {}
    for i in range(len(p2)):
        ser = p2.iloc[i]
        succeeded, info = get_info(ser["SMILES"])
        assert(succeeded is True)
        name = ser["Name"]

        p2_d[info.everything_hash] = name

    canonical_d = {}

    for i in range(len(canonical)):

        ser = canonical.iloc[i]
        succeeded, info = get_info(ser["SMILES"])
        name = ser["Name"]
        assert(succeeded is True)

        canonical_d[info.everything_hash] = name

    return p1_d, p2_d, canonical_d

   

def get_info(smiles, debug = False) -> SideChainInfo:

    try:
        mol = Chem.MolFromSmiles(smiles)
        assert(mol is not None)
    except Exception as e:
        return (False, "RDKit error")

    patterns = get_patterns()
    mol_graph, _ = graphs.make_graph(mol)
    if debug:
        graphs.draw_graph(mol_graph)
        for pattern in patterns:
            graphs.draw_graph(pattern.graph, draw_order = False)
        
    hits = []
    #find all possible matches of residue patterns
    for pattern in patterns:
        matcher = nx.algorithms.isomorphism.GraphMatcher(mol_graph, pattern.graph, node_match = utils.node_match)
        for _, subgraph in enumerate(matcher.subgraph_isomorphisms_iter()):

            inverted_subgraph = {v:k for k,v in subgraph.items()}
            id_val = tuple(inverted_subgraph.values())
            id_val = hash(id_val)

            residue = Residue(inverted_subgraph, pattern)
            hits.append(residue)
        
    if len(hits) > 1: #patterns can be symmetrical, check if they're all the same hit
        assert(len(set([x.name for x in hits])))
    if len(hits) == 0:
        return (False, "No pattern hits")
    
    res = hits[0]
    
    info = walk_side_chain(res, None, mol)
    return (True, info)

class SideChainAnnotator:

    def __init__(self):

        self.priority_1, self.priority_2, self.canonical = get_named_hashes()

    def assign_name(self, info: SideChainInfo):

        query_hash = info.everything_hash

        if query_hash in self.priority_1:
            return self.priority_1[query_hash]
        elif query_hash in self.priority_2:
            return self.priority_2[query_hash]
        else:
            return query_hash

    def is_canonical(self, info: SideChainInfo):

        if info.everything_hash in self.canonical:
            return True
        else:
            return False

    def get_side_chain_type(self, ser):

        name = ser["Side chain name"]
        info = ser["Side chain info"]

        if ser["Terminal cyclization"] or ser["Internal cyclization"]:
            return SIDE_CHAIN_TYPE.CYCLIZATION
        if "adjacent" in name.lower() or "terminal" in name.lower() or "cyclization" in name.lower():
            return SIDE_CHAIN_TYPE.CYCLIZATION


        canonical = self.is_canonical(info)

        if canonical is True:
            return SIDE_CHAIN_TYPE.CANONICAL
        else:
            return SIDE_CHAIN_TYPE.NONCANONICAL

        raise Exception("Mishandled side chain type detection")


    def get_graph_categories(self, ser):

        side_chain_graph = ser["Side chain info"].everything_graph
        s = {}

        s["cat Fatty Acid"] = self.is_fatty_acid(side_chain_graph)

        # beta-hydroxy acids (hydroxylation of the side chain)
        bh_acid_status = self.is_beta_hydroxy_acid(ser, side_chain_graph)
        if bh_acid_status == "Backbone beta hydroxy acid":
            s["cat Backbone Beta Hydroxy Acid"] = True
            s["cat Side Chain Beta Hydroxy Acid"] = False
            s["cat Beta Hydroxy Acid"] = True

        elif bh_acid_status == "Side chain beta hydroxy acid":
            s["cat Side Chain Beta Hydroxy Acid"] = True
            s["cat Backbone Beta Hydroxy Acid"] = False
            s["cat Beta Hydroxy Acid"] = True

        elif bh_acid_status == "Side chain beta hydroxy acid with non-single bond":
            s["cat Side Chain Beta Hydroxy Acid"] = False
            s["cat Backbone Beta Hydroxy Acid"] = False
            s["cat Beta Hydroxy Acid"] = False
            s["Beta Hydroxy Acid with non-single bond"] = True

        elif bh_acid_status is None:
            s["cat Beta Hydroxy Acid"] = False
            s["cat Backbone Beta Hydroxy Acid"] = False
            s["cat Side Chain Beta Hydroxy Acid"] = False

        else:
            raise Exception("Unhandled beta hydroxy acid type")

        # thiazol/oxazol(in)es (a165cdec, d1206967, e4d62e96, 5a2b7ffc, 07383190)
        is_thia_oxa = self.is_thiazole(side_chain_graph) or self.is_oxazole(side_chain_graph)
        s["cat Thia/Oxazole"] = is_thia_oxa


        # piperazates (N-N bond (65fdc36c))
        is_pip = self.is_piperazate(side_chain_graph)
        s["cat Piperazate"] = is_pip

        return pd.Series(s)


    @staticmethod
    def is_fatty_acid(side_chain_graph):

        from networkx.algorithms import isomorphism

        elements = set([nx.get_node_attributes(side_chain_graph, "element")[x] for x in side_chain_graph.nodes])

        for element in elements:
            if element not in ["C", "CX", "O", "PX", "DX"]:
                return False

        if has_aromatic_ring(side_chain_graph):
            return False

        pattern = nx.Graph()
        pattern.add_nodes_from([(1, {"element":"CX"}), #alpha
                                       (2, {"element":"C"}), #rest of side chain
                                       (3, {"element":"C"}), #rest of side chain
                                       (4, {"element":"C"}), #rest of side chain
                                       (5, {"element":"C"}), #rest of side chain
                                       (6, {"element":"C"})]) #rest of side chain

        pattern.add_edges_from([(1,2),
                               (2,3),
                               (3,4),
                               (4,5),
                               (5,6)])

        GM = isomorphism.GraphMatcher(side_chain_graph, pattern, node_match = utils.node_match)
        if GM.subgraph_is_isomorphic():
            return True

        return False

    @staticmethod
    def is_disulfide(side_chain_graph):

        from networkx.algorithms import isomorphism
        pattern = nx.Graph()

        pattern.add_nodes_from([(1, {"element":"CX"}),
                                (2, {"element":"C"}),
                                (3, {"element":"S"}),
                                (4, {"element":"S"}),
                                (5, {"element":"C"}),
                                (6, {"element":"C"}),
                                ])

        pattern.add_edges_from([(1,2),
                               (2,3),
                               (3,4),
                               (4,5),
                               (5,6)])

        GM = isomorphism.GraphMatcher(side_chain_graph, pattern, node_match = utils.node_match)
        if GM.is_isomorphic():
            return True

        return False

    @staticmethod
    def is_beta_hydroxy_acid(ser, side_chain_graph):

        #manually override the canonicals
        if "serine" in ser["Side chain name"].lower() or "threonine" in ser["Side chain name"].lower():
            return None

        if ser["Backbone type"] == BACKBONE_TYPE.BETA and ser["Proximal bond"] == BOND.ESTER:
            return "Backbone beta hydroxy acid"

        else:
            
            pattern = nx.Graph()

            pattern.add_nodes_from([(1, {"element":"CX"}), #alpha
                                           (2, {"element":"C"}), #beta
                                           (3, {"element":"O"})]) #hydroxy

            pattern.add_edges_from([(1,2),
                                   (2,3)])

            #make sure oxygen has only one bond, would imply a hydrogen is attached
            GM = isomorphism.GraphMatcher(side_chain_graph, pattern, node_match = utils.node_match)
            if GM.subgraph_is_isomorphic():
                for match in GM.subgraph_isomorphisms_iter():
                    inv = {v:k for k,v in match.items()}
                    o_atom = inv[3]
                    assert(side_chain_graph.nodes[o_atom]["element"] == "O")
                    o_edges = list(side_chain_graph.edges(o_atom))
                    if len(o_edges) == 1:
                        order = side_chain_graph.get_edge_data(o_edges[0][0], o_edges[0][1])["order"]
                        if order != "1":
                            return "Side chain beta hydroxy acid with non-single bond"
                        else:
                            return "Side chain beta hydroxy acid"
            return None

    @staticmethod
    def is_thiazole(side_chain_graph):
        
        sulfur_pattern = nx.Graph()

        sulfur_pattern.add_nodes_from([(1, {"element":"CA"}),
                                       (2, {"element":"C"}),
                                       (3, {"element":"S"}),
                                       (4, {"element":"C"}),
                                       (5, {"element":"N"})])

        sulfur_pattern.add_edges_from([(1,2),
                               (2,3),
                               (3,4),
                               (4,5),
                               (5,1)])

        GM = isomorphism.GraphMatcher(side_chain_graph, sulfur_pattern, node_match = utils.node_match)

        if GM.subgraph_is_isomorphic():
            return True
        return False

    @staticmethod
    def is_oxazole(side_chain_graph):

        oxygen_pattern = nx.Graph()
        oxygen_pattern.add_nodes_from([(1, {"element":"CA"}), 
                                       (2, {"element":"C"}), 
                                       (3, {"element":"O"}), 
                                       (4, {"element":"C"}),
                                       (5, {"element":"N"})])

        oxygen_pattern.add_edges_from([(1,2),
                               (2,3),
                               (3,4),
                               (4,5),
                               (5,1)])

        GM = isomorphism.GraphMatcher(side_chain_graph, oxygen_pattern, node_match = utils.node_match)
        if GM.subgraph_is_isomorphic():
            return True
        return False

    @staticmethod
    def is_piperazate(side_chain_graph):

        pattern = nx.Graph()
        pattern.add_nodes_from([(1, {"element":"N"}), 
                                       (2, {"element":"N"})])

        pattern.add_edges_from([(1,2)])

        GM = isomorphism.GraphMatcher(side_chain_graph, pattern, node_match = utils.node_match)
        if GM.subgraph_is_isomorphic():
            return True
        return False

class Curator:

    def __init__(self, filename = f"{ROOT_DIR}/data/manual_curation.csv"):

        manual_label_df = pd.read_csv(filename, sep = ",", quoting = 2, index_col = None)

        #split out non-thiopeptide category, it's handled separately
        self.non_thiopeptide_df = manual_label_df[manual_label_df["class"] == "non_thiopeptide"]
        self.manual_label_df = manual_label_df[manual_label_df["class"] != "non_thiopeptide"]

    def curate(self, molecule_ser):

        molecule_ser = self._curate_class(molecule_ser)
        molecule_ser = self._check_stereo_info(molecule_ser)

        return molecule_ser

    def _curate_class(self, ser):

        new_ser = copy.copy(ser)

        new_ser["Automatic class"] = ser["Class"]
        new_ser["Manual class"] = None

        mol_id = new_ser["Molecule ID"].split("_")[0]
        
        matches = self.manual_label_df[self.manual_label_df["id_val"].apply(lambda x: x.split(",")[0] in mol_id)]
        if len(matches) > 1:
            print(matches)
            raise Exception(f"Multiple matches for id: {mol_id}")
        if len(matches) == 1:

            print("ONE")

            match = matches.iloc[0]
            new_class = match["class"]

            old_class = ser["Class"]

            old_stem = "/".join(old_class.split("/")[:-1])
            new_stem = "/".join(new_class.split("/")[:-1])


            if new_class == old_class:
                #logging.info(f"Keeping class for {mol_id}: {old_class}")
                pass
            else:
                if old_stem == new_stem:
                    #logging.info(f"Keeping class for {mol_id}: {old_class}")
                    pass
                else:
                    #logging.info(f"Overriding class for {mol_id}: {old_class}->{new_class}")
                    new_ser["Manual class"] = new_class

        if new_ser["Manual class"] is not None:
            new_ser["Class"] = new_ser["Manual class"]
            new_ser["Simple class"] = utils.get_simple_class(new_ser)
            print(new_ser["Simple class"])

        return new_ser

    def _check_stereo_info(self, molecule_ser):

        ser = molecule_ser.copy()

        from rdkit.Chem.rdchem import StereoSpecified
        from rdkit.Chem.rdmolops import FindPotentialStereo

        mol = molecule_ser["ROMol"]

        stereoinfo = FindPotentialStereo(mol)

        missing = [x.specified == StereoSpecified.Unspecified for x in stereoinfo]

        missing_stereo = any(missing)

        ser["Has undefined stereo"] = missing_stereo

        return ser

class PeptideGraph:

    def __init__(self, mol):

        self.nodes = {}
        self.mol = mol
        self.used_atoms = set()
        self.proximal_end = None
        self.distal_end = None
        self.cycle = False

    def __len__(self):

        return len(self.nodes)

    def try_cycle_close(self):

        open_proximal = []
        open_distal = []

        for node in self.nodes.values():
            if node.proximal is None:
                open_proximal.append(node)
            if node.distal is None:
                open_distal.append(node)

        logging.debug(f"OPEN PROXIMAL: {open_proximal}")
        logging.debug(f"OPEN DISTAL: {open_distal}")

        hits = []
        for prox in open_proximal:
            for dist in open_distal:

                idx1 = prox.proximal_het_idx
                idx2 = dist.distal_het_carbon_idx
                bond = self.mol.GetBondBetweenAtoms(idx1, idx2)
                if bond:
                    hits.append((prox, dist))

        if len(hits) == 0:
            raise Exception("No candidates for closing cycle")
        elif len(hits) == 1:
            #hit = hits[0]

            # degree_map = {"A": 1,
            #              "B": 2,
            #              "G": 3,
            #              }
            # degrees = [degree_map[x.get_degree()] for x in hit]
            # distal_degree = degrees[0]

            # print("THIS CYCLE REWARD")
            # self.cycle_reward = (100 * (3 - distal_degree)) + 1
            prox.proximal = dist.hash_val
            dist.distal = prox.hash_val
            self.cycle = True
            return

        elif len(hits) > 1:
            raise Exception("Multiple ways to close cycle, unable to handle")

    def add_closing_residue(self, residue, proximal_idx_of_attachment, distal_idx_of_attachment):

        if self.__len__() == 0:
            raise ParsingException("Can't add closing residue to empty graph")

        if residue.hash_val in self.nodes:
            return 0

        # new_residue_atoms = set(residue.subgraph.values())
        used_residue_atoms = set()
        for used_residue in self.nodes.values():
            for used_atom_idx in used_residue.subgraph.values():
                used_residue_atoms.add(used_atom_idx)

        residue = copy.copy(residue)

        proximal_residue_of_attachment = self.nodes[proximal_idx_of_attachment]
        distal_residue_of_attachment = self.nodes[distal_idx_of_attachment]

        if proximal_residue_of_attachment.distal:
            raise Exception("Proximal residue is full")
        if distal_residue_of_attachment.proximal:
            raise Exception("Proximal residue is full")
        residue.distal = proximal_idx_of_attachment
        residue.proximal = distal_idx_of_attachment
        self.nodes[residue.hash_val] = residue
        proximal_residue_of_attachment.proximal = residue.hash_val
        distal_residue_of_attachment.distal = residue.hash_val
        self.proximal_end = None
        self.distal_end = None
        self.cycle = True

        return 1

    def get_proximal_neighbor(self, residue):

        if residue.proximal is None:
            return None
        proximal_neighbor = self.nodes[residue.proximal]
        return proximal_neighbor

    def get_distal_neighbor(self, residue):

        if residue.distal is None:
            return None
        distal_neighbor = self.nodes[residue.distal]
        return distal_neighbor

    def set_cycle(self, residue, direction):

        if direction == DIRECTION.PROXIMAL:
            if not residue.proximal:
                logging.debug("Called set_cycle with no proximal neighbor")
                return

            next_residue = residue.proximal
            while next_residue is not None:
                curr_residue = self.nodes[next_residue]
                curr_residue.on_cycle = True
                next_residue = curr_residue.proximal
            assert curr_residue == self.proximal_end

        elif direction == DIRECTION.DISTAL:

            if not residue.distal:
                logging.debug("Called set_cycle with no distal neighbor")
                return

            next_residue = residue.distal
            while next_residue is not None:
                curr_residue = self.nodes[next_residue]
                curr_residue.on_cycle = True
                next_residue = curr_residue.distal
            assert curr_residue == self.distal_end
        else:
            raise Exception("'direction' must be 'proximal' or 'distal'")

    def add_residue(self, residue, idx_of_attachment=None, end=None):
        # end must be "proximal" or "distal"

        residue = copy.copy(residue)
        if self.__len__() == 0:
            self.nodes[residue.hash_val] = residue
            self.proximal_end = residue
            self.distal_end = residue
            return 1

        if residue.hash_val in self.nodes:
            return 0

        # new_residue_atoms = set(residue.subgraph.values())
        used_residue_atoms = set()
        for used_residue in self.nodes.values():
            for used_atom_idx in used_residue.subgraph.values():
                used_residue_atoms.add(used_atom_idx)

        if False:
            pass
        else:
            residue_of_attachment = self.nodes[idx_of_attachment]
            if end == "proximal":
                if residue_of_attachment.proximal:
                    raise ParsingException("Proximal end is already filled")

                residue.distal = idx_of_attachment
                self.nodes[residue.hash_val] = residue
                residue_of_attachment.proximal = residue.hash_val
                self.proximal_end = residue
                return 1

            elif end == "distal":
                if residue_of_attachment.distal:
                    raise ParsingException("Distal end is already filled")

                residue.proximal = idx_of_attachment
                self.nodes[residue.hash_val] = residue
                residue_of_attachment.distal = residue.hash_val
                self.distal_end = residue
                return 1

            else:
                raise ParsingException("end must be 'proximal' or 'distal' if PeptideGraph is not empty")

    def summarize(self):
        print("=== PeptideGraph ===")
        [print(x) for x in self.nodes.values()]
        print(f"CYCLE: {self.cycle}")
        print("==================")

    def get_sequence(self):

        if self.cycle:
            # choose a starting point arbitrarily
            starting_node = list(self.nodes.values())[0]
            current_node = starting_node
            sequence = []

            while current_node is not None:
                sequence.append(current_node.side_chain_name)
                current_node = self.nodes[current_node.distal]
                if current_node == starting_node:
                    break

            return sequence

        else:
            current_node = self.proximal_end
            sequence = []

            while current_node is not None:
                sequence.append(current_node.side_chain_name)
                if current_node.distal is None:
                    break
                current_node = self.nodes[current_node.distal]
            return sequence

    def get_residues(self):

        return self.nodes.values()

    def get_backbone_carbons(self):

        carbons = []
        for residue in self.nodes.values():
            carbons.extend(residue.get_backbone_carbons())

        return carbons

    def get_all_atoms(self):

        atoms = []
        for residue in self.get_residues():
            atoms.extend(residue.subgraph.values())

        return atoms

    def get_distance(self, res1, res2):
        # return the sequence distance between the two residues

        curr_res = res1

        dist = 0
        found = False
        while True:
            if curr_res.distal is None:
                break
            next_res = self.nodes[curr_res.distal]
            dist += 1
            
            if next_res == res2:
                found = True
                break
            else:
                curr_res = next_res

        if not found:
            dist = 0
            curr_res = res1
            while True:
                if curr_res.proximal is None:
                    break
                next_res = self.nodes[curr_res.proximal]
                dist += 1
                
                if next_res == res2:
                    found = True
                    break
                else:
                    curr_res = next_res

        if not found:
            raise Exception("Can't get distance between residues")

        return dist
        # keep track of terminal residue atoms for checking cyclization points later

    def get_terminal_residue_backbone_atoms(self):

        terminal_residue_backbone_atoms = []
        for res in self.get_residues():
            if res.is_terminal():
                terminal_residue_backbone_atoms.extend(res.get_atom_indices())

        terminal_residue_backbone_atoms = set(terminal_residue_backbone_atoms)
        return terminal_residue_backbone_atoms

    def __contains__(self, subgraph):
        raise Exception("this doesn't work")

        hash_val = hash(tuple(subgraph.values()))
        if hash_val in self.nodes:
            for atom in subgraph.values():
                if atom in self.used_atoms:
                    return False
        else:
            return False

        return True

    # does not modify existing objects, returns a merged copy
    def merge(self, other):

        a = copy.copy(self)
        b = copy.copy(other)

        new_nodes = {}

        any_change = False
        for a_node_idx, a_node in a.nodes.items():
            for b_node_idx, b_node in b.nodes.items():
                if a_node_idx == b_node_idx:
                    if not a_node.proximal and not b_node.distal:
                        any_change = True
                        new_nodes[b_node_idx] = b.nodes[b_node_idx]
                        new_nodes[b_node_idx].distal = a_node_idx

                        new_nodes[a_node_idx] = a.nodes[a_node_idx]
                        new_nodes[a_node_idx].proximal = b_node_idx

                    elif not a_node.distal and not b_node.proximal:
                        any_change = True

                        new_nodes[b_node_idx] = b.nodes[b_node_idx]
                        new_nodes[b_node_idx].proximal = a_node_idx

                        new_nodes[a_node_idx] = a.nodes[a_node_idx]
                        new_nodes[a_node_idx].distal = b_node_idx

        if any_change:
            merged = PeptideGraph()
            merged.nodes = a.nodes
            merged.nodes.update(b.nodes)
            merged.nodes.update(new_nodes)
            merged.used_atoms = a.used_atoms.union(b.used_atoms)

            return merged

        else:
            return None


class Residue:

    def __init__(self, subgraph: dict, pattern):

        self.distal = None
        self.proximal = None
        self.other_distal = None
        self.name = pattern.name
        if pattern.name != "Pyridine":
            self.distal_het_carbon_idx = subgraph[pattern.distal_heteroatom_carbon]
            self.proximal_het_idx = subgraph[pattern.proximal_heteroatom]
            self.distal_het_idx = subgraph[pattern.distal_heteroatom]
            self.subgraph = subgraph
            hash_val = hash(tuple(subgraph.values()))
            self.hash_val = hash_val
            logging.debug(f"Residue got alpha: {pattern.alpha_carbon}")
            logging.debug(f"Residue got beta: {pattern.beta_carbon}")
            self.alpha = subgraph[pattern.alpha_carbon]
            if pattern.beta_carbon:
                self.beta = subgraph[pattern.beta_carbon]
            else:
                self.beta = None
            if pattern.gamma_carbon:
                self.gamma = subgraph[pattern.gamma_carbon]
            else:
                self.gamma = None

            self.weight = 101 - (len(self.get_backbone_carbons()) * 2)
            self.side_chain_name = None
            self.on_cycle = None
        else:
            self.distal_het_carbon_idx = None
            self.proximal_het_idx = None
            self.distal_het_idx = None
            self.subgraph = subgraph
            self.hash_val = hash(tuple(subgraph.values()))
            self.alpha = None
            self.beta = None
            self.gamma = None
            self.side_chain_name = None
            self.weight = 0

    def __repr__(self):
        return f"Residue(A:{self.alpha}|B:{self.beta}|G:{self.gamma})"

    def get_distal_connection_idx(self):
        if self.name != "Pyridine":

            return self.distal_het_carbon_idx
        else:
            val = self.subgraph[12]
            #print("PYRIDINE CNX IDX: ", val)
            return val

    def get_other_distal_connection_idx(self):
        if self.name != "Pyridine":
            raise Exception("Only pyridine has multiple distal connection points")

        val = self.subgraph[9]
        #print("PYRIDINE CNX IDX: ", val)
        return val

    def get_proximal_connection_idx(self):
        if self.name != "Pyridine":
            return self.proximal_het_idx
        else:
            val =  self.subgraph[8]
            #print("PYRIDINE CNX IDX: ", val)
            return val

    def get_backbone_carbons(self) -> List[int]:

        if self.name == "Pyridine":
            return []
        carbons = [self.alpha]
        if self.beta is not None:
            carbons.append(self.beta)
        if self.gamma is not None:
            carbons.append(self.gamma)

        return carbons

    def get_degree(self):

        if self.name == "Pyridine":
            return "P"
        elif self.beta == None and self.gamma == None:
            return "A"
        elif self.gamma == None:
            return "B"
        else:
            return "G"

    def __eq__(self, other):

        if self is None and other is None:
            return True
        elif self is None or other is None:
            return False

        if self.hash_val == other.hash_val:
            return True
        else:
            return False

    def get_atom_indices(self):
        return set(self.subgraph.values())

    def is_terminal(self):

        return self.is_n_terminus() or self.is_c_terminus()


    def is_n_terminus(self):
        if self.distal and not self.proximal:
            return True
        return False

    def is_c_terminus(self):

        if self.proximal and not self.distal:
            return True
        return False

    def __str__(self):
        return f"{str(self.hash_val)[:9]}, ALPHA: {self.alpha}, DISTAL: {self.distal}, PROX: {self.proximal}, SIDE: {self.side_chain_name}"

class Parser:

    def __init__(self):

        self.namer = SideChainAnnotator()
        self.accessibility_dict, _ = accessibility.parse_accessibility_database("data/accessibility_database.csv")
        
        self.patg_dict, _ = accessibility.parse_accessibility_database("data/patg_cyclization_database.csv")
        self.pcy1_dict, _ = accessibility.parse_accessibility_database("data/pcy1_cyclization_database.csv")

        self.curator = Curator()

    def post_annotation(self, ser, mol):

        old_ser = ser.copy()

        s = annotate_nitrogen_graphs(ser, mol)
        ser = pd.concat((ser,s))

        s = get_match_categories(ser)
        ser = pd.concat((ser,s))

        s = self.namer.get_graph_categories(ser)
        ser = pd.concat((ser,s))
        
        ser.name = old_ser.name

        side_chain_type = self.namer.get_side_chain_type(ser)

        ser["Side chain type"] = side_chain_type

        residue_type = get_residue_type(ser)
        ser["Residue type"] = residue_type

        residue_name = utils.get_residue_name(ser)
        ser["Residue name"] = residue_name


        return ser

    def parse_residue(self, residue: Residue, backbone: PeptideGraph, mol: RWMol):

        if residue.name == "Pyridine":
            print("SKIPPING PYRIDINE")
            return
       
        match_attributes = {}

        match_attributes["Internal cyclization"] = False
        match_attributes["Terminal cyclization"] = False

        side_chain_info = walk_side_chain(residue, backbone, mol)

        # R/S and D/L

        from macrocycles.common import LD

        def get_ld(residue, mol):
            if len(residue.get_backbone_carbons()) == 1:
                alpha_idx = residue.alpha

                side_chain_starts = []

                for atom in mol.GetAtomWithIdx(alpha_idx).GetNeighbors():
                    if atom.GetIdx() in residue.get_atom_indices():
                        continue
                    side_chain_starts.append(atom.GetIdx())

                if len(side_chain_starts) != 1:
                    return LD.OTHER
                else:
                    this_mol = copy.copy(mol)

                    this_mol.GetAtomWithIdx(side_chain_starts[0]).SetAtomicNum(32)
                    rdCIPLabeler.AssignCIPLabels(this_mol)

                    try:
                        label = this_mol.GetAtomWithIdx(alpha_idx).GetProp("_CIPCode")
                    except KeyError:
                        return LD.OTHER
                    if label == "S":
                        return LD.L
                    elif label == "R":
                        return LD.D
                    else:
                        return LD.OTHER

            else:
                return LD.OTHER

        ld = get_ld(residue, mol)

        match_attributes["LD"] = ld

        match_attributes["Side chain info"] = side_chain_info

        used_backbone_carbons = [x for x in side_chain_info.graph.nodes if side_chain_info.graph.nodes[x]["element"] == "CX"]

        match_attributes["Side chain graph"] = side_chain_info.full_backbone_graph

        side_chain_name = self.namer.assign_name(side_chain_info)

        match_attributes["Side chain name"] = side_chain_name

        residue.side_chain_name = side_chain_name
        match_attributes["Side chain hash"] = side_chain_info.side_chain_hash
        match_attributes["Orderless side chain hash"] = side_chain_info.orderless_side_chain_hash
        match_attributes["Orderless graph"] = side_chain_info.orderless_graph

        match_attributes["Full side chain graph"] = side_chain_info.full_backbone_graph
        match_attributes["Full side chain hash"] = side_chain_info.full_backbone_side_chain_hash

        match_attributes["Everything graph"] = side_chain_info.everything_graph
        match_attributes["Everything hash"] = side_chain_info.everything_hash

        match_attributes["Legacy graph"] = side_chain_info.legacy_graph
        match_attributes["Legacy hash"] = side_chain_info.legacy_hash

        #defaults
        match_attributes["Internal cyclization"] = False
        match_attributes["Internal cyclization distance"] = None
        match_attributes["Terminal cyclization"] = False
        match_attributes["Terminal cyclization distance"] = None
        match_attributes["Multiple internal cyclizations"] = False
        match_attributes["Multiple terminal cyclizations"] = False
        match_attributes["Internal and terminal cyclizations"] = False
        match_attributes["Loop"] = False

        events = side_chain_info.events
        event_types = [x.type for x in events]
        if EVENT_TYPE.SELF_LOOP in event_types:
            match_attributes["Loop"] = True
        else:
            match_attributes["Loop"] = True

        #after handling self loops, update events to remove self loop events
        events = [x for x in events if x.type != EVENT_TYPE.SELF_LOOP]

        if len(events) == 0:
            pass
        elif len(events) > 1:

            event_types = [x.type for x in events]

            events_to_parse = [x for x in events if x.type in [EVENT_TYPE.INTERNAL_CYCLIZATION, EVENT_TYPE.TERMINAL_CYCLIZATION]]
            events_to_parse = [x for x in events_to_parse if x.attributes["distance"] > 0]

            
            internal_cyclizations = [x for x in events_to_parse if x.type == EVENT_TYPE.INTERNAL_CYCLIZATION]
            terminal_cyclizations = [x for x in events_to_parse if x.type == EVENT_TYPE.TERMINAL_CYCLIZATION]

            if len(internal_cyclizations) > 0 and len(terminal_cyclizations) > 0:
                match_attributes["Internal and terminal cyclizations"] = True
            else:

                if len(internal_cyclizations) == 1:
                    match_attributes["Internal cyclization"] = True
                elif len(internal_cyclizations) > 1:
                    match_attributes["Multiple internal cyclizations"] = True

                if len(terminal_cyclizations) == 1:
                    match_attributes["Terminal cyclization"] = True
                elif len(terminal_cyclizations) > 1:
                    match_attributes["Multiple terminal cyclizations"] = True


            match_attributes["Internal cyclization distance"] = 0
            match_attributes["Terminal cyclization"] = False
            match_attributes["Terminal cyclization distance"] = 0
            match_attributes["Side chain name"] = "!Multiple_events"

        else:
            event = events[0]
        
            if event.type == EVENT_TYPE.SELF_LOOP:
                match_attributes["Internal cyclization"] = False
                match_attributes["Internal cyclization distance"] = 0
                match_attributes["Terminal cyclization"] = False
                match_attributes["Terminal cyclization distance"] = 0
                match_attributes["Loop"] = True

            elif event.type == EVENT_TYPE.TERMINAL_CYCLIZATION:
                match_attributes["Terminal cyclization"] = True
                match_attributes["Terminal cyclization distance"] = event.attributes["distance"]
                match_attributes["Terminal cyclization direction"] = event.attributes["target"]
                match_attributes["Internal cyclization"] = False
                match_attributes["Internal cyclization distance"] = 0
                match_attributes["Loop"] = False

            elif event.type == EVENT_TYPE.INTERNAL_CYCLIZATION:
                match_attributes["Internal cyclization"] = True
                match_attributes["Internal cyclization distance"] = event.attributes["distance"]
                match_attributes["Terminal cyclization"] = False
                match_attributes["Terminal cyclization distance"] = 0
                match_attributes["Loop"] = False

            else:
                raise Exception(f"Unhandled event type: {event.type}")

        visited_atoms = list(side_chain_info.graph.nodes)

        #special case for glycine
        if match_attributes["Side chain name"] == "Glycine":
            match_attributes["Side chain atom indices"] = residue.get_backbone_carbons()
        else:

            ##!!!!!!
            ## 2023-04-12: this is now all atoms in side chain, including unused backbone carbons
            ##!!!!!!
            match_attributes["Side chain atom indices"] = visited_atoms

        match_attributes["Backbone atom indices"] = residue.get_atom_indices()

        # check whether own heteroatom comes from peptide or ester
        proximal_atom = mol.GetAtomWithIdx(residue.proximal_het_idx)
        if proximal_atom.GetSymbol() == "N":
            match_attributes["Backbone nitrogen to oxygen"] = False
        elif proximal_atom.GetSymbol() == "O":
            match_attributes["Backbone nitrogen to oxygen"] = True

        # check whether backbone is peptide or ester
        distal_symbol = mol.GetAtomWithIdx(residue.distal_het_idx).GetSymbol()
        if distal_symbol == "N":
            bond_type = BOND.PEPTIDE
        elif distal_symbol == "O":
            bond_type = BOND.ESTER
        else:
            raise Exception("This statement shouldn't be reachable")

        distal_bond_type = bond_type

        #determine proximal heteroatom substituents
        proximal_heteroatom = mol.GetAtomWithIdx(residue.proximal_het_idx)
        neighbors = proximal_heteroatom.GetNeighbors()
        neighbor_count = len(neighbors)
        neighbor_symbols = [x.GetSymbol() for x in neighbors]

        if neighbor_count <= 1:
            bond_type = "Terminal"
        if neighbor_count > 3:
            raise ParsingException("Proximal heteroatom has more than three neighbors")
        else: # must be two or three neighbors

            start_symbol = proximal_heteroatom.GetSymbol()
            if start_symbol == "N":
                bond_type = BOND.PEPTIDE
            elif start_symbol == "O":
                bond_type = BOND.ESTER
            else:
                raise ParsingException("Unhandled proximal heteroatom type in bond detection")

            if neighbor_count == 3: # nitrogen probably has some modification

                neighbor_idxs = [x.GetIdx() for x in neighbors]
                proximal_neighbor = backbone.get_proximal_neighbor(residue)

                # remove the normal atoms bound to nitrogen:
                # side chain carbon
                # carbon of "carbonyl" on proximal side
                # if there's no proximal neighbor, we have nothing to remove there
                # could be a terminal residue but a modified nitrogen
                if proximal_neighbor:
                    unaccounted_idxs = [x for x in neighbor_idxs if x not in match_attributes["Side chain atom indices"] and x != proximal_neighbor.distal_het_carbon_idx]
                else:
                    unaccounted_idxs = [x for x in neighbor_idxs if x not in match_attributes["Side chain atom indices"]]

                if len(unaccounted_idxs) == 0: #no modification
                    match_attributes["Backbone nitrogen substituents"] = None
                elif len(unaccounted_idxs) == 1:
                    off_cycle_neighbor = unaccounted_idxs[0]

                    #walk along nitrogen modification and keep track
                    atoms_for_walk = set()
                    visited_atoms = [residue.proximal_het_idx]
                    to_visit = [off_cycle_neighbor]

                    loop = False
                    while len(to_visit) > 0:
                        atom_idx = to_visit.pop(-1)
                        if atom_idx in backbone.get_all_atoms():
                            visited_atoms.append(atom_idx)
                            loop = True
                            continue
                        atom = mol.GetAtomWithIdx(atom_idx)
                        for neighbor in atom.GetNeighbors():
                            neighbor_idx = neighbor.GetIdx()
                            if neighbor_idx in visited_atoms or neighbor_idx in to_visit:
                                continue
                            to_visit.append(neighbor_idx)
                        visited_atoms.append(atom_idx)
                    match_attributes["Backbone nitrogen substituents"] = visited_atoms
                else:
                    match_attributes["Backbone nitrogen substituents"] = "Multiple"
                    #raise ParsingException("Proximal heteroatom has too many unaccounted neighbors to parse")

        proximal_bond_type = bond_type

        if not proximal_bond_type and distal_bond_type:
            raise ParsingException("Missing bond type detection")

        match_attributes["Proximal bond"] = proximal_bond_type
        match_attributes["Distal bond"] = distal_bond_type

        match_attributes["N-terminus"] = residue.is_n_terminus()
        match_attributes["C-terminus"] = residue.is_c_terminus()
        

        if match_attributes["Terminal cyclization"] or match_attributes["Internal cyclization"]:

            stereo = STEREO.FUSION

        else:

            # print(match_attributes["Side chain name"])
            # print(used_backbone_carbons)
            if match_attributes["Side chain name"] == "Glycine":
                stereo = STEREO.NO_SUBSTITUENTS
            elif len(used_backbone_carbons) >= 1:
                #atom = mol.GetAtomWithIdx(used_backbone_carbons[0])
                atom = mol.GetAtomWithIdx(residue.get_backbone_carbons()[0])
                try:
                    stereo = atom.GetProp("_CIPCode")
                except:
                    stereo = "Unspecified"

                # find just the backbone carbons that are used in the side chain
                # e.g. if gamma, and there's a 'valine' coming off the alpha
                # carbon, we still record stereochemistry
                backbone_carbons = []
                for node in side_chain_info.graph.nodes:
                    if side_chain_info.graph.nodes[node]["element"] == "CX":
                        backbone_carbons.append(node)

                if len(set(backbone_carbons)) > 1:
                    stereo = STEREO.MULTIPLE_BACKBONE_CARBONS
                else:


                    if stereo == "R":
                        stereo = STEREO.R
                    elif stereo == "S":
                        stereo = STEREO.S
                    elif stereo == "Unspecified":

                        backbone_atoms = match_attributes["Backbone atom indices"]
                        backbone_carbon = backbone_carbons[0]
                        assert(backbone_carbon in backbone_atoms)
                        
                        # find number of non-backbone atoms coming off the backbone carbon
                        edges = []
                        for edge in side_chain_info.graph.edges:
                            if edge[0] in backbone_atoms and edge[1] in backbone_atoms: # ignore backbone-to-backbone edges
                                continue
                            if edge[0] == backbone_carbon  or edge[1] == backbone_carbon:
                                edges.append(edge)

                        if len(edges) == 0:
                            stereo = STEREO.NO_SUBSTITUENTS
                        elif len(edges) > 1:
                            stereo = STEREO.MULTIPLE_SUBSTITUENTS
                        else: # check if single substituent of backbone carbon is a double bond
                            edge = edges[0]     
                            order = side_chain_info.graph.edges[edge]["order"]
                            if order not in ["1"]:
                                stereo = STEREO.NOT_STEREOCENTER
                            else:
                                stereo = STEREO.MISSING

                    else:
                        raise Exception("Unhandled stereo")
            else:
                raise Exception("Unhandled stereo 2")

        if stereo not in [STEREO.R,
                          STEREO.S,
                          STEREO.MISSING,
                          STEREO.NO_SUBSTITUENTS,
                          STEREO.NOT_STEREOCENTER,
                          STEREO.MULTIPLE_BACKBONE_CARBONS,
                          STEREO.MULTIPLE_SUBSTITUENTS,
                          STEREO.FUSION]:
            raise Exception("Unhandled stereo")

        match_attributes["Side chain carbon stereo"] = stereo

        if residue.gamma:
            match_attributes["Backbone type"] = BACKBONE_TYPE.GAMMA
        elif residue.beta:
            match_attributes["Backbone type"] = BACKBONE_TYPE.BETA
        else:
            match_attributes["Backbone type"] = BACKBONE_TYPE.ALPHA

        if "Backbone nitrogen substituents" not in match_attributes:
            match_attributes["Backbone nitrogen substituents"] = None

        ser = pd.Series(match_attributes)
        return ser


    def parse_mol(self, mol, verbose = False, id_val = "dummy"):
        molecule_attributes, match_attributes = self._parse(mol, verbose, id_val)

        molecule_ser = pd.Series(molecule_attributes)
        match_df = pd.DataFrame(match_attributes)

        a = [molecule_ser["Molecule ID"]] * len(match_df)
        b = list(range(len(match_df)))


        multiindex = pd.MultiIndex.from_arrays([a, b])
        match_df.index = multiindex
        match_df["Molecule ID"] = a 

        if pd.notnull(molecule_ser["Reason for exclusion"]):
            molecule_ser["Class"] = "excluded"
            return (molecule_ser, match_df)

        annotated = []
        for i in range(len(match_df)):
            match_ser = match_df.iloc[i].copy()
            s = self.post_annotation(match_ser, mol)
            annotated.append(s)

        match_df = pd.DataFrame(annotated)
        match_df.index = pd.MultiIndex.from_tuples(match_df.index)

        tier = utils.get_tier(molecule_ser, match_df)
        molecule_ser["Class"] = tier


        
        molecule_ser = self.curator.curate(molecule_ser)

        molecule_ser["Simple class"] = utils.get_simple_class(molecule_ser)

        if molecule_ser["Class"].split("/")[0] == "peptide":
            molecule_ser, match_df = accessibility.annotate_accessibility(molecule_ser, match_df, self.accessibility_dict, self.patg_dict, self.pcy1_dict)

        return molecule_ser, match_df

    def _parse(self, mol, verbose = False, id_val = None):

        molecule_attributes = {}

        if type(id_val) == tuple:
            id_string = "_".join(id_val)
        else:
            id_string = id_val

        molecule_attributes["Molecule ID"] = id_string


        molecule_attributes["Inchi"] = Chem.MolToInchi(mol)
        molecule_attributes["Smiles"] = Chem.MolToSmiles(mol)
        molecule_attributes["ROMol"] = mol
        molecule_attributes.update(utils.get_rdkit_descriptors(mol))

        failure = None

        backbone, parsing_info = get_backbone(mol, molecule_name = id_val)

        rdCIPLabeler.AssignCIPLabels(mol)


        if not backbone:
            failure = "Did not find a peptide backbone"
        
        if backbone and backbone.cycle:
            molecule_attributes["Cyclic backbone"] = True
        else:
            molecule_attributes["Cyclic backbone"] = False
        if failure:
            molecule_attributes["Reason for exclusion"] = failure
            return molecule_attributes, []


        for k, v in parsing_info.items():
            molecule_attributes[k] = v
        molecule_attributes["Reason for exclusion"] = None

        residue_matches = []
        terminal_fusions = []
        
        if len(backbone) <= 2:
            molecule_attributes["Reason for exclusion"] = "Two or fewer residues detected"
            return molecule_attributes, []

        for residue in backbone.get_residues():

            try:
                this_match_attributes = self.parse_residue(residue, backbone, mol)
                residue_matches.append((residue, this_match_attributes))
            except Exception as e:
                print(e)
                traceback.print_exc()
                molecule_attributes["Reason for exclusion"] = f"Error during residue parsing: {str(e)}"
                break

        if molecule_attributes["Reason for exclusion"]:
            return molecule_attributes, []

        molecule_attributes["Backbone indices"] = backbone.get_all_atoms()

        terminal_fusions = [x for x in residue_matches if x[1]["Terminal cyclization"]]

        if len(terminal_fusions) == 1:
            residue, match_attributes = terminal_fusions[0]
            dist = match_attributes["Terminal cyclization distance"]
            direction = match_attributes["Terminal cyclization direction"]
            backbone.set_cycle(residue, direction)

        matched_backbone_atoms = set()
        all_matched_atoms = set()
        for _, match_attributes in residue_matches:
            a = match_attributes["Backbone atom indices"]
            backbone_nitrogen_substituents = match_attributes["Backbone nitrogen substituents"]
            if backbone_nitrogen_substituents:
                all_matched_atoms = all_matched_atoms.union(backbone_nitrogen_substituents)
            matched_backbone_atoms = matched_backbone_atoms.union(a)
            all_matched_atoms = all_matched_atoms.union(match_attributes["Backbone atom indices"]).union(match_attributes["Side chain atom indices"])

        all_matched_atoms = all_matched_atoms.union(backbone.get_all_atoms())
        prop = len(matched_backbone_atoms.intersection(backbone.get_all_atoms())) / len(backbone.get_all_atoms())

        molecule_attributes["Backbone indices"] = backbone.get_all_atoms()
        molecule_attributes["Proportion of all atoms matched"] = len(all_matched_atoms) / mol.GetNumAtoms()
        molecule_attributes["All matched atoms"] = all_matched_atoms
        molecule_attributes["Side chain sequence"] = backbone.get_sequence()

        all_match_attributes = [x[1] for x in residue_matches]

        molecule_attributes["Has Disulfide"] = any(["Disulfide" in x for x in all_match_attributes])

        #every match is on cycle if backbone is a perfect cycle
        if backbone.cycle:
            for _ , match_attributes in residue_matches:
                match_attributes["On main cycle"] = True
        else: #query individual matches for cycle status
            assert len(backbone) == len(residue_matches)
            for i, residue in enumerate(backbone.get_residues()):
                _, match_attributes = residue_matches[i]
                if residue.on_cycle is not None:
                    match_attributes["On main cycle"] = residue.on_cycle
                else:
                    match_attributes["On main cycle"] = False
        return (molecule_attributes, all_match_attributes)

    def parse_smiles(self, smiles):

        mol = Chem.MolFromSmiles(smiles)
        return self.parse_mol(mol)
    

class MultipleGraph():

    #residues are allowed to have multiple connections
    #checks if residues are actually connected in molecule when constructing graph

    def __init__(self, mol):

        self.mol = mol
        self.nodes = []

    def add_residue(self, residue):

        used_residue_atoms = set()
        for used_residue in [x.residue for x in self.nodes]:
            for used_atom_idx in used_residue.subgraph.values():
                used_residue_atoms.add(used_atom_idx)

        logging.debug("START ADDING RESIDUE")
        new_node = MultipleNode(residue)
        if len(self.nodes) == 0:
            self.nodes.append(new_node)
        else:
            added = False
            for other_node in self.nodes:
                other_residue = other_node.residue

                res_atoms = residue.get_atom_indices()
                other_res_atoms = other_residue.get_atom_indices()

                shared_atoms = res_atoms.intersection(other_res_atoms)
                if len(shared_atoms) > 1:
                    continue

                
                idx1 = residue.get_distal_connection_idx()
                idx2 = other_residue.get_proximal_connection_idx()

                bond = self.mol.GetBondBetweenAtoms(idx1, idx2)
                if bond:
             
                    new_residue_atoms = set(residue.subgraph.values())
                    new_node.add_proximal_node(other_node)
                    other_node.add_distal_node(new_node)

                idx1 = other_residue.get_distal_connection_idx()
                idx2 = residue.get_proximal_connection_idx()

                bond = self.mol.GetBondBetweenAtoms(idx1, idx2)
                if bond:
 
                    new_residue_atoms = set(residue.subgraph.values())
                    new_node.add_distal_node(other_node)
                    other_node.add_proximal_node(new_node)

                
            self.nodes.append(new_node)

    def get_nx_graph(self):

        g = nx.DiGraph()

        labels = {}
        for node in self.nodes:
            name = node.residue.hash_val
            name = node.residue.hash_val
            labels[node.residue.hash_val] = node.residue.hash_val
            logging.debug(f"GETTING ALPHA: {node.residue.alpha}")
            logging.debug(f"GETTING BETA: {node.residue.beta}")
            logging.debug(f"GETTING BACKBONE CARBONS: {node.residue.get_backbone_carbons()}")
            backbone_carbons = ",".join([str(x) for x in node.residue.get_backbone_carbons()])
            degree = node.residue.get_degree()
            label = node.residue.get_degree() + " " + backbone_carbons
            g.add_node(name, weight = node.weight, backbone_carbons = backbone_carbons, degree = degree, label = label)

        for node in self.nodes:
            for distal_node in node.distal_nodes:
                g.add_edge(node.residue.hash_val, distal_node.residue.hash_val)

        return g

    def draw(self, filename = None):

        import matplotlib.pyplot as plt

        plt.figure(figsize = (20, 20))
        g = self.get_nx_graph()

        layout = nx.fruchterman_reingold_layout(g)
        labels = nx.get_node_attributes(g, "label")
        logging.debug(labels)
        colormap = []


        colors = []
        for key, label in labels.items():

            if label.strip() == "P":
                colors.append("#88ff88")
            else:
                colors.append("#8888ff")


        nx.draw_networkx(g, pos = layout, node_size = 1200, arrows = True, arrowstyle = "-|>", arrowsize = 50, labels = labels, node_color = colors)

        if filename:
            plt.savefig(filename, dpi = 200)

        else:
            plt.show()
        plt.close()

    def best_path(self, molecule_name = None):

        parsing_info = {}

        global_graph = self.get_nx_graph()

        # hash_val is hash of all atom indices in residue
        # node_dict = {hash_val: Residue}
        node_dict = {}
        for node in global_graph.nodes():
            hits = [x for x in self.nodes if x.residue.hash_val == node]
            assert(len(hits) == 1)
            node_dict[node] = hits[0].residue

        paths_for_scoring = []
        all_paths = []
        for start_node in global_graph:
            paths = []
            closed_paths = []
            # logging.debug(f"DFS START NODE: {global_graph.nodes[start_node]['backbone_carbons']}")
            to_visit = [start_node]
            visited_nodes = set()
            
            while True:

                to_visit.sort(key = lambda x:len(global_graph.nodes[x]['backbone_carbons']), reverse = True)

                try:
                    current_node = to_visit.pop()
                except:
                    break

                if current_node in visited_nodes:
                    logging.debug("\tALREADY VISITED")
                    continue

                new_paths = []

                # get actual Residue object
                current_residue: Residue = node_dict[current_node]
                node_found = False
                for path in paths:
                    #logging.debug(f"TRYING TO ADD {current_residue.get_backbone_carbons()} to {path}")
                    
                    #if current node is connected to end of path
                    if current_node in global_graph.neighbors(path[-1]):

                        #do not add a node if any of its alpha carbons are already used
                        query_alphas = set(current_residue.get_backbone_carbons())
                        existing_alphas = path.get_backbone_carbons()
                        if len(existing_alphas.intersection(query_alphas)) > 0:
                            #logging.debug("\tALPHA OVERLAP, SKIPPING")
                            continue

                        #do not add a node if more than one of its atoms are already used
                        # TODO: could be formal about which atoms are connected, haven't seen an error yet though
                        proposed_atoms = node_dict[current_node].get_atom_indices()
                        used_atoms = [node_dict[x].get_atom_indices() for x in path.nodes]
                        s = set()
                        for x in used_atoms:
                            [s.add(i) for i in x]
                        used_atoms = s

                        if len(proposed_atoms.intersection(used_atoms)) > 2:
                            #logging.debug(f"CANT ADD RESIDUE {current_residue.get_backbone_carbons} to path {path} due to too many atom overlaps")
                            continue

                        #create a new path with current node appended to it and add to pile of possible paths
                        node_found = True
                        new_path = path.copy()
                        new_path.append(node_dict[current_node])
                        #logging.debug(f"SUCCESSFULLY ADDED {current_residue.get_backbone_carbons()} to {new_path}")
                        new_paths.append(new_path)


                paths.extend(new_paths)

                #if node can't be added to any existing paths, create a new path containing it alone
                if not node_found:
                    paths.append(Path(self.mol, global_graph, [node_dict[current_node]]))

                #consider current node visited and add its neighbors to queue for visiting
                visited_nodes.add(current_node)
                for neighbor in global_graph.neighbors(current_node):
                    to_visit.append(neighbor)

            all_paths.extend(paths)
            all_paths.extend(closed_paths)


        #paths = paths + closed_paths
        paths = all_paths
        [logging.debug(f"CANDIDATE PATH: {path}") for path in paths]
        for path in paths:
            logging.debug(f"CYCLE: {path.cycle}")

            paths_for_scoring.extend(paths)

        final_path_set = set()
        final_paths = []

        for path in paths_for_scoring:
            #final_path_set.add(path.get_backbone_carbons())
            carbons = path.get_backbone_carbons()
            key = tuple(sorted(carbons))
            if key not in final_path_set:
                final_path_set.add(key)
                final_paths.append(path)

        logging.info(f"{final_paths}")

        #try to close path into cycle if possible
        for path in final_paths:
            for current_node in path:
                for existing_node in path:
                    if existing_node in global_graph.neighbors(current_node):
                        start = node_dict[current_node].get_backbone_carbons()
                        end = node_dict[existing_node].get_backbone_carbons()
                        try:
                            new_path = copy.copy(path)
                            new_path.add_cycle_edge((node_dict[current_node], node_dict[existing_node]))
                            final_paths.append(new_path)
                            #logging.info(f"FINAL SUCCEED ADDING CYCLE EDGE from {start} to {end} in {path}")
                            logging.debug(f"IS CYCLE: {new_path.cycle}")
                            
                        except Exception as e:
                            #logging.info(f"FINAL FAILED ADDING CYCLE EDGE from {start} to {end} in {path}: {e}")
                            pass

        paths_for_scoring = final_paths

        for path in paths_for_scoring:

            score, positive_score, score_dict = path.get_value()
            score_dict["len"] = len(path)
            score_dict["weight"] = score
            score_dict["nodes"] = [global_graph.nodes[x]['backbone_carbons'] for x in path.nodes]
            import json
            logging.info(f"PATH: {json.dumps(score_dict)}")

        if len(paths_for_scoring) == 0:
            raise Exception("No best paths to choose from")

        max_length = max([len(x) for x in paths_for_scoring])
        max_cycles = [x for x in paths_for_scoring if x.cycle and len(x) == max_length]
        max_paths = [x for x in paths_for_scoring if len(x) == max_length]

        best_path_just_reward = None
        best_path_with_penalty = None

        for use_penalty in [False, True]:

            best_value = 0
            best_paths = []
            for path in paths_for_scoring:
                score, positive_score, score_dict = path.get_value()
                if use_penalty:
                    value = score
                else:
                    value = positive_score

                if value == best_value:
                    best_paths.append(path)
                elif value > best_value:
                    best_paths = [path]
                    best_value = value

            if use_penalty:
                penalty_text = " WITH PENALTY"
            else:
                penalty_text = ""
            logging.debug(f"OVERALL BEST VALUE{penalty_text}: {best_value}")
            logging.debug(f"OVERALL BEST LENGTHs{penalty_text}: {[len(x) for x in best_paths]}")
            logging.debug(f"OVERALL BEST PATHS{penalty_text}: {best_paths}")
            logging.debug(f"OVERALL BEST PATH IS CYCLE: {best_paths[0].cycle}")

            if use_penalty:
                if len(best_paths) == 0:
                    best_path_with_penalty = None
                else:
                    best_path_with_penalty = best_paths[0]
            else:
                if len(best_paths) == 0:
                    best_path_just_reward = None
                else:
                    best_path_just_reward = best_paths[0]

        if best_path_with_penalty == None and best_path_just_reward == None:
            parsing_info["Backbone path has penalty"] = False
        elif best_path_with_penalty != best_path_just_reward:
            parsing_info["Backbone path has penalty"] = True
        else:
            parsing_info["Backbone path has penalty"] = False



        #construct PeptideGraph() for best path
        if len(best_paths) == 0:
            return (None, None)

        path = best_paths[0]
        pg = PeptideGraph(self.mol)
        last_idx = None
        for node in path:
            residues = [x.residue for x in self.nodes if x.residue.hash_val == node]
            if len(residues) != 1:
                raise Exception("Multiple matches for residue hash?")
            residue = residues[0]
            if last_idx:
                try:
                    retval = pg.add_residue(residue, idx_of_attachment = last_idx, end = 'proximal')
                except:
                    pass

            else:
                retval = pg.add_residue(residue)

            if retval == 0:
                raise Exception("ADDING FAILED")

            last_idx = residue.hash_val

        try:
            pg.try_cycle_close()
        except Exception as e:
            pass

        return pg, parsing_info

class Path:

    def __init__(self, mol, parent_graph, residues = None, edges = None):

        self.mol = mol
        self.parent_graph = parent_graph
        self.cycle = False

        if residues is None:
            self.residues = []
            self.nodes = []
            self.edges = []
        else:
            self.residues = residues
            self.nodes = [x.hash_val for x in self.residues]
            if edges is None:
                self.edges = []
            else:
                self.edges = edges

    def get_nx_graph(self):

        g = nx.DiGraph()

        labels = {}
        for i, node in enumerate(self.nodes):
            g.add_node(node)
            #labels[node] = ",".join([str(x) for x in self.residues[i].get_backbone_carbons()])
            labels[node] = ",".join([str(x) for x in self.residues[i].get_backbone_carbons()])

        for start, end in self.edges:
            g.add_edge(start, end)

        return g


    def draw(self, filename = None):

        import matplotlib.pyplot as plt

        g = nx.DiGraph()

        labels = {}
        for i, node in enumerate(self.nodes):
            g.add_node(node)
            #labels[node] = ",".join([str(x) for x in self.residues[i].get_backbone_carbons()])
            labels[node] = ",".join([str(x) for x in self.residues[i].get_backbone_carbons()])

        for start, end in self.edges:
            g.add_edge(start, end)

        nx.draw_networkx(g, labels = labels)

        if filename:
            plt.savefig(filename, dpi = 200)
        else:
            plt.show()

        plt.close()

    def append(self, residue):

        self.edges.append((self.nodes[-1], residue.hash_val))
        node = residue.hash_val
        self.residues.append(residue)
        self.nodes.append(node)

    def get_backbone_carbons(self):

        carbons = set()
        for residue in self.residues:
            for idx in residue.get_backbone_carbons():
                carbons.add(idx)

        return carbons

    def add_cycle_edge(self, edge):

        hit = edge
        edge = tuple([x.hash_val for x in edge])
        existing_ends = [x[1] for x in self.edges]
        if edge[1] in existing_ends:
            raise Exception("Node already connected to")

        degree_map = {"A": 1,
                      "B": 2,
                      "G": 3,
                      }
        degrees = [degree_map[x.get_degree()] for x in hit]
        distal_degree = degrees[0]
        degree_set = set(degrees)
        if degree_set == set([1]):
            self.cycle_reward = 95
        else:
            self.cycle_reward = 93
        
        logging.debug(f"CYCLE REWARD: {degrees}, {self.cycle_reward}")

        #self.cycle_reward = (100 * (3 - distal_degree)) + 1

        #print(f"ACTUAL CYCLE REWARD: {self.cycle_reward}")

        self.edges.append(edge)
        self.cycle = True


    def __getitem__(self, idx):

        return self.nodes[idx]

    def __len__(self):

        return len(self.nodes)

    def get_value(self):

        score_dict = {}

        for node in self.nodes:
            node_val = nx.get_node_attributes(self.parent_graph, "weight")[node]
            degree = nx.get_node_attributes(self.parent_graph, "degree")[node]

            if degree not in score_dict:
                score_dict[degree] = 0 

            score_dict[degree] += node_val

        if self.cycle:
            #logging.debug("IS BACKBONE CYCLE")
            if self.cycle_reward > 0:
                logging.debug(f"ADDING CYCLE REWARD OF {self.cycle_reward}")
            score_dict["Backbone cycle"] = self.cycle_reward

        side_chain_cycle = self.detect_side_chain_cyclization()

        if side_chain_cycle:

            score_dict["Side chain cycle"] = 94

        invalid_cyclization = self.detect_invalid_cyclization()
        
        if invalid_cyclization:
            score_dict["Invalid cyclization"] = -400

        #terminal oxygen is bad
        if not self.cycle:
            for i in [-1]: #check both ends
                res = self.residues[i]
                if res.name == "Pyridine":
                    continue
                proximal_heteroatom_idx = res.proximal_het_idx
                proximal_heteroatom = self.mol.GetAtomWithIdx(proximal_heteroatom_idx)
                element_symbol = proximal_heteroatom.GetSymbol()
                if element_symbol == "O":
                    num_neighbors = len(proximal_heteroatom.GetNeighbors())
                    if num_neighbors <= 1:
                        logging.debug("APPLYING PUNISHMENT")
                        #penalty += 100
                        if "Terminal oxygen" not in score_dict:
                            score_dict["Terminal oxygen"] = 0
                        score_dict["Terminal oxygen"] -= 50

        total_score = 0
        positive_score = 0
        for value in score_dict.values():
            total_score += value
            if value > 0:
                positive_score += value

        return total_score, positive_score, score_dict

    def old_get_value(self):

        reward = self.get_reward()
        penalty = self.get_penalty()
            
        return reward - penalty

    def get_reward(self):

        side_chain_cycle = self.detect_side_chain_cyclization()

        invalid_cyclization = self.detect_invalid_cyclization()

        reward = 0

        for node in self.nodes:
            node_val = nx.get_node_attributes(self.parent_graph, "weight")[node]
            reward += node_val

        if self.cycle:
            self.has_cycle = True
            logging.debug("IS BACKBONE CYCLE")
            if self.cycle_reward > 0:
                logging.debug(f"ADDING CYCLE REWARD OF {self.cycle_reward}")
            reward += self.cycle_reward
        else:
            self.has_cycle = False

        if side_chain_cycle:
            logging.debug(f"ADDING SIDE CHAIN CYCLE REWARD")
            self.has_side_chain_cycle = True
            #reward += 200 # works for most of test_list
            #reward += 199
            reward += 94
        else:
            self.has_side_chain_cycle = False

        if invalid_cyclization:
            self.has_invalid_cyclization = True
            logging.debug(f"ADDING INVALID CYCLE PENALTY")
            reward -= 400
        else:
            self.has_invalid_cyclization = True
    
        return reward

    def detect_invalid_cyclization(self):

        for residue in self.residues:

            idx_map = {}
            for other_residue in self.residues:
                if residue == other_residue:
                    continue

                for atom in other_residue.get_atom_indices():
                    idx_map[atom] = other_residue


            starts = []
            own_atoms = residue.get_atom_indices()

            for backbone_carbon_idx in residue.get_backbone_carbons():

                backbone_carbon = self.mol.GetAtomWithIdx(backbone_carbon_idx)
                if len(backbone_carbon.GetNeighbors()) == 2: #has no side chain
                    continue

                for neighbor in backbone_carbon.GetNeighbors():
                    if neighbor.GetIdx() not in own_atoms:
                        starts.append(neighbor.GetIdx())

            #side chain walk loop
            visited_atoms = []
            to_visit = starts
            while len(to_visit) > 0:
                atom_idx = to_visit.pop(-1)
                logging.debug(f"START INVALID CHECK FROM {atom_idx}")
                
                if atom_idx in own_atoms:
                    visited_atoms.append(atom_idx)

                #side chain looped onto self, can safely ignore and continue traversal elsewhere
                if atom_idx in visited_atoms:
                    logging.debug("ALREADY VISITED")
                    continue

                direct_edges = []

                #if this atom belongs to another residue
                if atom_idx in idx_map:

                    for backbone_carbon_idx in residue.get_backbone_carbons():
                        for neighbor in backbone_carbon.GetNeighbors():
                            direct_edges.append(neighbor.GetIdx())

                if atom_idx in direct_edges:
                    if atom_idx in own_atoms:
                        logging.debug(f"Would skip {atom_idx} but it's in this residue")
                    else:
                        logging.debug(f"Skipping immediate neighbor {atom_idx}")
                        logging.debug(f"BACKBONE CARBONS {residue.get_backbone_carbons()}")
                        logging.debug(f"OWN ATOMS: {own_atoms}")
                        return True

                #add neighbors to walk list
                atom = self.mol.GetAtomWithIdx(atom_idx)
                for neighbor in atom.GetNeighbors():
                    neighbor_idx = neighbor.GetIdx()
                    if neighbor_idx in visited_atoms or neighbor_idx in to_visit:
                        continue
                    to_visit.append(neighbor_idx)

                visited_atoms.append(atom_idx)


        return False


    def detect_side_chain_cyclization(self):

        if len(self.residues) < 4:
            return False

        for residue in self.residues:



            idx_map = {}
            for other_residue in self.residues:
                if residue == other_residue:
                    continue

                for atom in other_residue.get_atom_indices():
                    idx_map[atom] = other_residue


            starts = []
            own_atoms = residue.get_atom_indices()

            for backbone_carbon_idx in residue.get_backbone_carbons():

                backbone_carbon = self.mol.GetAtomWithIdx(backbone_carbon_idx)
                if len(backbone_carbon.GetNeighbors()) == 2: #has no side chain
                    continue

                for neighbor in backbone_carbon.GetNeighbors():
                    if neighbor.GetIdx() not in own_atoms:
                        starts.append(neighbor.GetIdx())

            #side chain walk loop
            visited_atoms = []
            to_visit = starts
            while len(to_visit) > 0:
                atom_idx = to_visit.pop(-1)
                
                if atom_idx in own_atoms:
                    visited_atoms.append(atom_idx)


                #side chain looped onto self, can safely ignore and continue traversal elsewhere
                if atom_idx in visited_atoms:
                    logging.debug("ALREADY VISITED")
                    continue

                if atom_idx in idx_map:


                    direct_edges = []
                    for backbone_carbon_idx in residue.get_backbone_carbons():
                        for neighbor in backbone_carbon.GetNeighbors():
                            logging.debug(f"Skipping immediate neighbor {atom_idx}")
                            direct_edges.append(neighbor.GetIdx())

                    if atom_idx in direct_edges:
                        continue
                    encountered_idx = self.residues.index(idx_map[atom_idx])
                    own_idx = self.residues.index(residue)

                    seq_distance = abs(encountered_idx - own_idx)

                    own_backbone_carbons = residue.get_backbone_carbons()
                    encountered_backbone_carbons = idx_map[atom_idx].get_backbone_carbons()

                    if seq_distance >= 3:
                        #print("ENC: ", atom_idx)
                        logging.debug(f"Found loop of dist {seq_distance} from {own_backbone_carbons} to {encountered_backbone_carbons}")
                        logging.debug(f"Atoms visited already: {len(visited_atoms)}")
                        return True
                    else:
                        logging.debug(f"Found loop of dist {seq_distance} from {own_backbone_carbons} to {encountered_backbone_carbons}, ignoring as it's too small")
                        continue

                #add neighbors to walk list
                atom = self.mol.GetAtomWithIdx(atom_idx)
                for neighbor in atom.GetNeighbors():
                    neighbor_idx = neighbor.GetIdx()
                    if neighbor_idx in visited_atoms or neighbor_idx in to_visit:
                        continue
                    to_visit.append(neighbor_idx)

                visited_atoms.append(atom_idx)

        return False
    def get_penalty(self):

        penalty = 0
        if not self.cycle:
            for i in [-1]: #check both ends
                res = self.residues[i]
                if res.name == "Pyridine":
                    continue
                proximal_heteroatom_idx = res.proximal_het_idx
                proximal_heteroatom = self.mol.GetAtomWithIdx(proximal_heteroatom_idx)
                element_symbol = proximal_heteroatom.GetSymbol()
                if element_symbol == "O":
                    num_neighbors = len(proximal_heteroatom.GetNeighbors())
                    if num_neighbors <= 1:
                        logging.debug("APPLYING PUNISHMENT")
                        #penalty += 100
                        penalty += 50

        return penalty

    def copy(self):

        new_path = Path(self.mol, self.parent_graph, copy.copy(self.residues), copy.copy(self.edges))
        return new_path

    def __str__(self):

        s = ""
        for residue in self.residues:
            s += ","
            s += str(residue.get_backbone_carbons())

        return s

    def __repr__(self):
        return self.__str__()

class MultipleNode:

    def __init__(self, residue):

        self.residue = residue
        self.weight = self.residue.weight
        self.proximal_nodes = []
        self.distal_nodes = []

    def add_proximal_node(self, other):

        self.proximal_nodes.append(other)

    def add_distal_node(self, other):

        self.distal_nodes.append(other)

class ResiduePattern:

    def __init__(self, graph, name, alpha_carbon, beta_carbon, gamma_carbon, proximal_heteroatom, distal_heteroatom, distal_heteroatom_carbon):

        #utils.draw_graph(graph)
        self.graph = graph
        self.name = name
        self.alpha_carbon = alpha_carbon
        self.beta_carbon = beta_carbon
        self.gamma_carbon = gamma_carbon
        self.distal_heteroatom = distal_heteroatom
        self.distal_heteroatom_carbon = distal_heteroatom_carbon
        self.proximal_heteroatom = proximal_heteroatom

def get_patterns():

    alpha_backbone = nx.Graph()
    alpha_backbone.add_nodes_from([(1, {"element":"C"}), #distal heteroatom's carbon
                                   (2, {"element":"C"}), #alpha
                                   (3, {"element":["O","N"]}), #proximal_heteroatom
                                   (5, {"element":["O","S"]}), 
                                   (6, {"element":["O","N"]})]) #distal heteroatom

    alpha_backbone.add_edges_from([(1,6),
                                   (1,5),
                                   (1,2),
                                   (2,3)])

    alpha_pattern = ResiduePattern(alpha_backbone, "A Glycine", alpha_carbon = 2, beta_carbon = None, gamma_carbon =None, proximal_heteroatom = 3, distal_heteroatom = 6, distal_heteroatom_carbon = 1)

    beta_backbone = nx.Graph()
    beta_backbone.add_nodes_from([(1, {"element":"C"}), #distal heteroatom's carbon
                                   (2, {"element":"C"}), #alpha
                                   (3, {"element":["O","N"]}), #proximal_heteroatom
                                   (5, {"element":["O","S"]}), 
                                   (6, {"element":["O","N"]}), #distal heteroatom
                                   (7, {"element":"C"})])

    beta_backbone.add_edges_from([(1,6),
                                   (1,5),
                                   (1,7),
                                   (2,7),
                                   (2,3)])

    beta_pattern = ResiduePattern(beta_backbone, "B Glycine", alpha_carbon = 2, beta_carbon = 7, gamma_carbon = None, proximal_heteroatom = 3, distal_heteroatom = 6, distal_heteroatom_carbon = 1)

    gamma_backbone = nx.Graph()
    gamma_backbone.add_nodes_from([(1, {"element":"C"}), #distal heteroatom's carbon
                                   (2, {"element":"C"}), #alpha
                                   (3, {"element":["O","N"]}), #proximal_heteroatom
                                   (5, {"element":["O","S"]}), 
                                   (6, {"element":["O","N"]}), #distal heteroatom
                                   (7, {"element":"C"}),
                                   (8, {"element":"C"})])

    gamma_backbone.add_edges_from([(1,6),
                                   (1,5),
                                   (1,8),
                                   (2,7),
                                   (7,8),
                                   (2,3)])

    gamma_pattern = ResiduePattern(gamma_backbone, "G Glycine", alpha_carbon = 2, beta_carbon = 7, gamma_carbon = 8, proximal_heteroatom = 3, distal_heteroatom = 6, distal_heteroatom_carbon = 1)

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


    #graphs.draw_graph(pyridine_graph, draw_order = False)
    pyridine_pattern = ResiduePattern(pyridine_graph, "Pyridine", alpha_carbon = None, beta_carbon = None, gamma_carbon = None, proximal_heteroatom = None, distal_heteroatom = None, distal_heteroatom_carbon = None)



    #patterns = [alpha_pattern, beta_pattern, gamma_pattern, pyridine_pattern]
    patterns = [alpha_pattern, beta_pattern, gamma_pattern]
    return patterns

def get_backbone(mol, verbose = False, timeout = 5, molecule_name = None):

    mol_graph, _ = graphs.make_graph(mol)
    patterns = get_patterns()

    def node_match(node1, node2):

        if node1["element"] == "R" or node2["element"] == "R":
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

    all_matches = []

    #find all possible matches of residue patterns
    for pattern in patterns:
        matcher = nx.algorithms.isomorphism.GraphMatcher(mol_graph, pattern.graph, node_match = node_match)
        for _, subgraph in enumerate(matcher.subgraph_isomorphisms_iter()):

            inverted_subgraph = {v:k for k,v in subgraph.items()}
            id_val = tuple(inverted_subgraph.values())
            id_val = hash(id_val)

            residue = Residue(inverted_subgraph, pattern)
            all_matches.append(residue)

    for residue in all_matches:
        if residue.name == "Pyridine":
            print(f"PYRIDINE: {residue.get_atom_indices()}")
        logging.debug(f"ALPHA OF DETECTED RESIDUE: {residue.alpha}")
        logging.debug(f"BETA OF DETECTED RESIDUE: {residue.beta}")

    #construct a graph of all possible paths
    graph = MultipleGraph(mol)
    for residue in all_matches:
        graph.add_residue(residue)

    #if molecule_name:
    #    graph.draw(filename = f"{molecule_name}_backbone_graph.svg")
        #graph.draw(filename = f"{molecule_name}_backbone_graph.png")

    #prune graph to best single path
    best_graph, parsing_info = graph.best_path(molecule_name)

    return best_graph, parsing_info


def walk_side_chain(residue: Residue, backbone: PeptideGraph, mol: RWMol):

    if backbone:
        other_terminal_backbone_atoms: set[int] = backbone.get_terminal_residue_backbone_atoms() \
                                            - set(residue.get_atom_indices())

        other_residue_backbone_atoms: set[int] = set(backbone.get_all_atoms()) - backbone.get_terminal_residue_backbone_atoms() - set(residue.get_atom_indices())

        logging.debug(f"TERMINAL: {other_terminal_backbone_atoms}")
        logging.debug(f"OTHER: , {other_terminal_backbone_atoms}")

        neighboring_residue_backbone_atoms: set[int] = set()
        if residue.proximal:
            atoms_to_add = backbone.nodes[residue.proximal].subgraph.values()
            neighboring_residue_backbone_atoms = neighboring_residue_backbone_atoms.union(atoms_to_add)
        if residue.distal:
            atoms_to_add = backbone.nodes[residue.distal].subgraph.values()
            neighboring_residue_backbone_atoms = neighboring_residue_backbone_atoms.union(atoms_to_add)


    starting_carbons = []
    #print("ATOM_IND: ", residue.get_atom_indices())
    #start with only subsituted backbone carbons
    for atom_idx in residue.get_backbone_carbons():
        neighbors = set([x.GetIdx() for x in mol.GetAtomWithIdx(atom_idx).GetNeighbors()])
        #print(atom_idx, [x.GetIdx() for x in neighbors])
        # print(set(neighbors) - set(residue.get_atom_indices()))
        if len(set(neighbors) - set(residue.get_atom_indices())) > 0:
            starting_carbons.append(atom_idx)

    # print("STARTING CARBONS: ", starting_carbons)



    #side chain walk loop
    #visited_atoms = residue.get_backbone_carbons()
    visited_atoms = copy.copy(starting_carbons)
    #visited_atoms = []

    #initially populate atoms to visit:
    to_visit = []
    for atom_idx in starting_carbons:
        atom = mol.GetAtomWithIdx(atom_idx)
        for neighbor in atom.GetNeighbors():
            neighbor_idx = neighbor.GetIdx()
            #if neighbor_idx in visited_atoms or neighbor_idx in to_visit:
            if neighbor_idx in residue.get_atom_indices():
                continue
            to_visit.append(neighbor_idx)

    # print("INITIAL TV: ", to_visit)


    ignore_internal = False
    cyclization_distance = None

    events = []
    while len(to_visit) > 0:

        # print("TV: ", to_visit)
        atom_idx = to_visit.pop(-1)
        # print("AIX: ", atom_idx)
        #logging.debug(f"VISITING: {atom_idx}")

        if atom_idx in residue.get_backbone_carbons():
            continue

        if atom_idx == residue.proximal_het_idx:
            logging.debug("IS PROXIMAL HET")
            visited_atoms.append(atom_idx)
            events.append(Event(EVENT_TYPE.SELF_LOOP, atom_idx, {"location":"self_proximal_het"}))
            continue
        elif atom_idx in residue.get_atom_indices():
            visited_atoms.append(atom_idx)
            events.append(Event(EVENT_TYPE.SELF_LOOP, atom_idx, {"location":"self"}))
            continue

        #######################################################
        # handle side chain looping onto immediate neighbors
        #######################################################
        if backbone:
            proximal_neighbor = backbone.get_proximal_neighbor(residue)
            distal_neighbor = backbone.get_distal_neighbor(residue)

            if proximal_neighbor and atom_idx == proximal_neighbor.distal_het_idx:
                logging.debug("IS PROXIMAL NEIGHBOR HET")
                visited_atoms.append(proximal_neighbor.distal_het_idx)
                events.append(Event(EVENT_TYPE.SELF_LOOP, atom_idx, {"location":"proximal_neighbor_het"}))
                continue
            elif proximal_neighbor and atom_idx in proximal_neighbor.get_atom_indices():
                logging.debug("IS IN PROXIMAL NEIGHBOR BACKBONE")
                if mol.GetAtomWithIdx(atom_idx).GetSymbol() == "C":
                    logging.info("Encountered proximal backbone carbon, likely internal cyclization")
                    events.append(Event(EVENT_TYPE.INTERNAL_CYCLIZATION, atom_idx, {"distance": 0}))
                    continue
                else:
                    visited_atoms.append(proximal_neighbor.distal_het_carbon_idx)
                    visited_atoms.append(residue.proximal_het_idx)
                    visited_atoms.append(atom_idx)
                    logging.info("Adding proximal neighbor's distal het carbon to complete loop")
                    events.append(Event(EVENT_TYPE.SELF_LOOP, atom_idx, {"location":"distal_neighbor_het"}))
                    continue
            elif distal_neighbor and atom_idx in distal_neighbor.get_atom_indices():
                logging.debug("IS IN DISTAL NEIGHBOR BACKBONE")
                logging.info("Side chain loops onto distal neighbor, likely internal cyclicization")
                logging.debug(visited_atoms)
                events.append(Event(EVENT_TYPE.INTERNAL_CYCLIZATION, atom_idx, {"distance":0}))
                visited_atoms.append(atom_idx)
                continue

            ###################################################################################
            # handle side chain looping onto other non-terminal residues
            # should only reach here if we haven't found a self loop or immediate neighbor loop
            ###################################################################################
            if atom_idx in other_residue_backbone_atoms and not atom_idx in other_terminal_backbone_atoms:

                dist = None

                #find the residue we've encountered
                for check_residue in backbone.get_residues():
                    if residue == check_residue:
                        continue
                    if atom_idx in check_residue.get_atom_indices():
                        if dist:
                            raise Exception("Setting internal loop distance twice?")
                        dist = backbone.get_distance(residue, check_residue)
                if dist is None:
                    raise Exception("Can't determine where internal cyclization lands")
                event = Event(EVENT_TYPE.INTERNAL_CYCLIZATION, atom_idx, {"distance":dist})
                events.append(event)
                visited_atoms.append(atom_idx)
                continue

            ###################################################################################
            # handle side chain looping onto other terminal residues
            # should only reach here if we haven't found a self loop,
            # immediate neighbor loop, or internal loop
            ###################################################################################
            elif atom_idx in other_terminal_backbone_atoms:
                visited_atoms.append(atom_idx)

                #every backbone should have a proximal and distal end if we've gotten to this point? 
                #lsp complains so I should make it clear in the code somewhere
                if atom_idx in backbone.proximal_end.get_atom_indices():
                    dist = backbone.get_distance(residue, backbone.proximal_end)
                    event = Event(EVENT_TYPE.TERMINAL_CYCLIZATION, atom_idx, {"distance": dist, "target":DIRECTION.PROXIMAL})
                    events.append(event)
                elif atom_idx in backbone.distal_end.get_atom_indices():
                    dist = backbone.get_distance(residue, backbone.distal_end)
                    event = Event(EVENT_TYPE.TERMINAL_CYCLIZATION, atom_idx, {"distance": dist, "target":DIRECTION.DISTAL})
                    events.append(event)
                else:
                    raise ParsingException("terminal side chain fusion detected but cannot identify which end")

                continue

        #side chain looped onto self, can safely ignore and continue traversal elsewhere
        #not an EVENT_TYPE.SELF_LOOP, that's only for backbone loops
        #maybe should be renamed
        #print("V: ", visited_atoms)
        if atom_idx in visited_atoms:
            continue

        #add neighbors to walk list
        atom = mol.GetAtomWithIdx(atom_idx)
        for neighbor in atom.GetNeighbors():
            neighbor_idx = neighbor.GetIdx()
            if neighbor_idx in visited_atoms or neighbor_idx in to_visit:
                continue
            to_visit.append(neighbor_idx)

        visited_atoms.append(atom_idx)

    legacy_visited_atoms = visited_atoms.copy()
    visited_atoms.extend(residue.get_backbone_carbons())
    generic_visited_atoms = copy.copy(visited_atoms)
    #generic_visited_atoms.append(residue.get_distal_connection_idx())
    #generic_visited_atoms.append(residue.get_proximal_connection_idx())

    #record side chain graph and name
    this_backbone_carbons = residue.get_backbone_carbons()

    legacy_side_chain_graph, legacy_side_chain_hash = \
        graphs.make_legacy_graph(mol, legacy_visited_atoms,
                          backbone_c_wildcards=this_backbone_carbons,
                          use_bond_order=True)

    side_chain_graph, side_chain_hash = \
        graphs.make_graph(mol, visited_atoms,
                          backbone_c_wildcards=this_backbone_carbons,
                          use_bond_order=True)

    orderless_graph, orderless_side_chain_hash = \
        graphs.make_graph(mol, visited_atoms,
                          backbone_c_wildcards=this_backbone_carbons,
                          use_bond_order=False)

    full_backbone_graph, full_hash = \
        graphs.make_graph(mol, generic_visited_atoms,
                          backbone_c_wildcards=this_backbone_carbons,
                          use_bond_order=True,
                          prox_idx = residue.get_proximal_connection_idx(),
                          dist_idx = residue.get_distal_connection_idx())

    with_stereo_graph, with_stereo_hash = \
        graphs.make_graph(mol, generic_visited_atoms,
                          backbone_c_wildcards=this_backbone_carbons,
                          use_bond_order=False,
                          prox_idx = residue.get_proximal_connection_idx(),
                          dist_idx = residue.get_distal_connection_idx(),
                          use_stereo = True)

    everything_graph, everything_hash = \
        graphs.make_graph(mol, generic_visited_atoms,
                          backbone_c_wildcards=this_backbone_carbons,
                          use_bond_order=True,
                          prox_idx = residue.get_proximal_connection_idx(),
                          dist_idx = residue.get_distal_connection_idx(),
                          use_stereo = True)



    #graphs.draw_graph(side_chain_graph)
    #graphs.draw_graph(full_backbone_graph)


    # print("FN: ", visited_atoms)
    # print("GN: ", generic_visited_atoms)
    # side_chain_name = assign_side_chain_name(residue, visited_atoms, side_chain_hash)

    # print(events)



    return SideChainInfo(graph = side_chain_graph,
                         side_chain_hash = side_chain_hash,
                         orderless_side_chain_hash = orderless_side_chain_hash,
                         orderless_graph = orderless_graph,
                         full_backbone_graph=full_backbone_graph,
                         full_backbone_side_chain_hash=full_hash,
                         with_stereo_hash=with_stereo_hash,
                         with_stereo_graph=with_stereo_graph,
                         legacy_graph = legacy_side_chain_graph,
                         legacy_hash = legacy_side_chain_hash,
                         everything_graph = everything_graph,
                         everything_hash = everything_hash,
                         events = events)
def annotate_nitrogen_graphs(ser, mol):
    
    s = {}

    nitrogen_substituents = ser["Backbone nitrogen substituents"]

    if type(nitrogen_substituents) == list or (nitrogen_substituents and type(nitrogen_substituents) != str and not np.isnan(nitrogen_substituents)):
        nitrogen_graph, hash_val = graphs.make_graph(mol, nitrogen_substituents)
        is_n_mod_oxygen = graph_is_oxygen_n_mod(nitrogen_graph)
        is_n_methyl = graph_is_n_methyl(nitrogen_graph)
    else:
        is_n_mod_oxygen = False
        is_n_methyl = False
    
    s["Nitrogen mod oxygen"] = is_n_mod_oxygen
    s["Graph n-methyl"] = is_n_methyl

    return pd.Series(s)

def get_match_categories(ser):

    s = {}

    # alpha amino acids

    if ser["LD"] == LD.D:
        s["cat D"] = True
    else:
        s["cat D"] = False

    if ser["Backbone type"] == BACKBONE_TYPE.BETA:
        s["cat Beta"] = True
    else:
        s["cat Beta"] = False

    if ser["Backbone type"] == BACKBONE_TYPE.GAMMA:
        s["cat Gamma"] = True
    else:
        s["cat Gamma"] = False

    # just alpha hydroxyacids
    if ser["Proximal bond"] == BOND.ESTER and ser["Backbone type"] == BACKBONE_TYPE.ALPHA:
        s["cat Hydroxy Acid"] = True
    else:
        s["cat Hydroxy Acid"] = False

    s["cat Stereo"] = get_side_chain_stereo(ser)

    # n methyl
    s["cat N-methyl"] = ser["Graph n-methyl"]

    if ser["Backbone nitrogen substituents"] and \
     (type(ser["Backbone nitrogen substituents"]) == list or pd.notnull(ser["Backbone nitrogen substituents"])) and \
     not ser["Graph n-methyl"]:
        s["cat Non-methyl nitrogen modification"] = True
    else:
        s["cat Non-methyl nitrogen modification"] = False

    return pd.Series(s)

def get_side_chain_stereo(s):
    
    from macrocycles.common import SIDE_CHAIN_STEREO
    if s["Backbone type"] != BACKBONE_TYPE.ALPHA:
        return SIDE_CHAIN_STEREO.NON_ALPHA
    elif s["Side chain carbon stereo"] == STEREO.R:
        return SIDE_CHAIN_STEREO.ALPHA_R
    elif s["Side chain carbon stereo"] == STEREO.S:
        return SIDE_CHAIN_STEREO.ALPHA_S
    else: 
        return SIDE_CHAIN_STEREO.OTHER

def graph_is_oxygen_n_mod(graph):
    
    pattern = nx.Graph()
    pattern.add_nodes_from([(1, {"element":"N"}),
                             (2, {"element":"O"})])
    pattern.add_edges_from([(1,2)])
    
    GM = isomorphism.GraphMatcher(graph, pattern, node_match = utils.node_match)
    if GM.is_isomorphic():
        return True
    else:
        return False

def graph_is_n_methyl(graph):
    
    pattern = nx.Graph()
    pattern.add_nodes_from([(1, {"element":"N"}),
                             (2, {"element":"C"})])
    pattern.add_edges_from([(1,2)])
    
    GM = isomorphism.GraphMatcher(graph, pattern, node_match = utils.node_match)
    if GM.is_isomorphic():
        return True
    else:
        return False

def has_aromatic_ring(side_chain_graph):

    five_pattern = nx.Graph()
    five_pattern.add_nodes_from([(1, {"element":"C"}), 
                                   (2, {"element":"C"}), 
                                   (3, {"element":"C"}), 
                                   (4, {"element":"C"}), 
                                   (5, {"element":"C"})]) 
    five_pattern.add_edges_from([(1,2, {"order":"12"}),
                                (2,3, {"order":"12"}),
                                (3,4, {"order":"12"}),
                                (4,5, {"order":"12"}),
                                (5,1, {"order":"12"})],)

    six_pattern = nx.Graph()
    six_pattern.add_nodes_from([(1, {"element":"C"}), 
                               (2, {"element":"C"}), 
                               (3, {"element":"C"}), 
                               (4, {"element":"C"}), 
                               (5, {"element":"C"}), 
                               (6, {"element":"C"})]) 
    six_pattern.add_edges_from([(1,2, {"order":"12"}),
                                (2,3, {"order":"12"}),
                                (3,4, {"order":"12"}),
                                (4,5, {"order":"12"}),
                                (5,6, {"order":"12"}),
                                (6,1, {"order":"12"})])

    GM = isomorphism.GraphMatcher(side_chain_graph, five_pattern, node_match = utils.node_match)
    if GM.subgraph_is_isomorphic():
        return True

    GM = isomorphism.GraphMatcher(side_chain_graph, six_pattern, node_match = utils.node_match)
    if GM.subgraph_is_isomorphic():
        return True

    return False

def get_residue_type(ser):

    assert(type(ser) == pd.Series)

    if ser["Side chain type"] == SIDE_CHAIN_TYPE.CYCLIZATION:
        return RESIDUE_TYPE.CYCLIZATION

    if ser["cat N-methyl"]:
        return RESIDUE_TYPE.NONCANONICAL
    if ser["cat Non-methyl nitrogen modification"]:
        return RESIDUE_TYPE.NONCANONICAL
    if ser["Proximal bond"] == BOND.ESTER:
        return RESIDUE_TYPE.NONCANONICAL

    if ser["Side chain type"] == SIDE_CHAIN_TYPE.NONCANONICAL:
        return RESIDUE_TYPE.NONCANONICAL
    elif ser["Side chain type"] == SIDE_CHAIN_TYPE.CANONICAL:
        return RESIDUE_TYPE.CANONICAL
    else:
        raise Exception("Can't determine residue type")


