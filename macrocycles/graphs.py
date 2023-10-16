import matplotlib.pyplot as plt
import networkx as nx
#from networkx.algorithms.graph_hashing import weisfeiler_lehman_graph_hash as wl_hash
from networkx import weisfeiler_lehman_graph_hash as wl_hash
from rdkit.Chem import rdCIPLabeler


def make_legacy_graph(mol, atom_indices = None, c_alpha_idx = None, c_beta_idx = None, backbone_c_wildcards = None, use_bond_order = True, compute_hash = True, prox_idx = None, use_stereo = False, dist_idx = None):

    rdCIPLabeler.AssignCIPLabels(mol)

    legacy_g = nx.Graph()

    if atom_indices == None:
        atom_indices = []
        for atom in mol.GetAtoms():
            atom_indices.append(atom.GetIdx())


    for atom_idx in atom_indices:
        element = mol.GetAtomWithIdx(atom_idx).GetSymbol()

        if backbone_c_wildcards and atom_idx in backbone_c_wildcards:
            element = "CA"

        elif atom_idx == dist_idx: #don't know where this would occur
            raise Exception("Distal carbon was encountered in side chain walk?")

        legacy_g.add_node(atom_idx, element = element)

    for bond in mol.GetBonds():
        end_idx = bond.GetEndAtom().GetIdx()
        start_idx = bond.GetBeginAtom().GetIdx()
        bond_order = str(int(bond.GetBondType()))

        if start_idx in legacy_g.nodes and end_idx in legacy_g.nodes:
            if use_bond_order:
                order = bond_order
            else:
                order = "1"
            legacy_g.add_edge(start_idx, end_idx, order = order)

    hash_val = "#" + wl_hash(legacy_g, edge_attr="order", node_attr="element")

    return legacy_g, hash_val

def make_graph(mol, atom_indices = None, c_alpha_idx = None, c_beta_idx = None, backbone_c_wildcards = None, use_bond_order = True, compute_hash = True, prox_idx = None, dist_idx = None, use_stereo = False):

    rdCIPLabeler.AssignCIPLabels(mol)

    g = nx.Graph()
    legacy_g = nx.Graph()
    #g_for_hash = nx.Graph()

    if atom_indices == None:
        atom_indices = []
        for atom in mol.GetAtoms():
            atom_indices.append(atom.GetIdx())


    for atom_idx in atom_indices:
        element = mol.GetAtomWithIdx(atom_idx).GetSymbol()

        is_aromatic = mol.GetAtomWithIdx(atom_idx).GetIsAromatic()
        hcount = mol.GetAtomWithIdx(atom_idx).GetTotalNumHs()

        #if atom_idx == c_alpha_idx:
        #    element = "CA"
        #elif atom_idx == c_beta_idx:
        #    element = "CB"
        if backbone_c_wildcards and atom_idx in backbone_c_wildcards:
            element = "CX"

        elif atom_idx == prox_idx: #special proline case
            element = element + "|PX"

        elif atom_idx == dist_idx: #don't know where this would occur
            raise Exception("Distal carbon was encountered in side chain walk?")

        atom = mol.GetAtomWithIdx(atom_idx)
        try:
            stereo = atom.GetProp("_CIPCode")
        except:
            stereo = "None"

        if stereo == "None":
            element_with_stereo_value = element
        else:
            element_with_stereo_value = f"{element}_{stereo}"

        g.add_node(atom_idx, element = element, stereo = stereo, element_with_stereo = element_with_stereo_value, aromatic = is_aromatic, hcount = hcount)
        if element == "CX":
            legacy_element = "CA"
        else:
            legacy_element = element
        legacy_g.add_node(atom_idx, element = element)

    if prox_idx is not None and prox_idx not in atom_indices:
        g.add_node(prox_idx, element = "PX", element_with_stereo = "PX", aromatic = is_aromatic)
    if dist_idx is not None and dist_idx not in atom_indices:
        g.add_node(dist_idx, element = "DX", element_with_stereo = "DX", aromatic = is_aromatic)


    for bond in mol.GetBonds():
        end_idx = bond.GetEndAtom().GetIdx()
        start_idx = bond.GetBeginAtom().GetIdx()
        bond_order = str(int(bond.GetBondType()))

        if start_idx in g.nodes and end_idx in g.nodes:
            if use_bond_order:
                order = bond_order
            else:
                order = "1"
            g.add_edge(start_idx, end_idx, order = order)

        if start_idx in legacy_g.nodes and end_idx in legacy_g.nodes:
            if use_bond_order:
                order = bond_order
            else:
                order = "1"
            legacy_g.add_edge(start_idx, end_idx, order = order)

    if compute_hash:
        if use_stereo:
            attr_key = "element_with_stereo"
        else:
            attr_key = "element"

        hash_val = "#" + wl_hash(g, edge_attr="order", node_attr=attr_key)
    else:
        hash_val = None

    return g, hash_val

def draw_graph(graph, filename = None, label = None, draw_indices = False, ax = None, draw_order = True, use_stereo = False, return_pil = True):

    #layout = nx.spring_layout(graph, scale = 0.8)
    layout = nx.kamada_kawai_layout(graph, scale = 0.8)
    #layout = nx.planar_layout(graph)
    labels = {}
    colormap = []


    figsize = (8,3)

    plt.figure(figsize = figsize)

    if draw_order:
        edge_labels = {}

        for edge in graph.edges.data():
            order = edge[2]['order']
            if order == '12':
                edge_labels[(edge[0], edge[1])] = "Aromatic"
            elif order == '2':
                edge_labels[(edge[0], edge[1])] = "Double"
            elif order == '3':
                edge_labels[(edge[0], edge[1])] = "Triple"

    for node, attributes in graph.nodes.data():

        if draw_indices:
            try:
                labels[node] = str(node) + ":" + attributes["element"]
            except:
                labels[node] = str(node) + ":" + "|".join(attributes["element"])
        else:

            try:
                labels[node] = attributes["element"]
            except:
                labels[node] = "|".join(attributes["element"])

        if use_stereo:
            try:
                if attributes["stereo"] != "None":
                    labels[node] += f"[{attributes['stereo']}]"
            except:
                pass

        if attributes["element"] == "C":
            colormap.append("green")
        elif attributes["element"] == "N":
            colormap.append("blue")
        elif attributes["element"] == "O":
            colormap.append("red")
        elif attributes["element"] in ["CA", "CB", "CG", "CX"]:
            colormap.append("pink")
        elif attributes["element"] == "S":
            colormap.append("yellow")
        else:
            colormap.append("grey")

    if ax:
        nx.rescale_layout_dict(layout, scale = 1)
        nx.draw(graph, pos = layout, node_color = colormap, node_size = [1000] * len(colormap), ax = ax)
        #nx.draw_networkx_nodes(graph, pos = layout, node_color = colormap, node_size = [100] * len(colormap), ax = ax, margins = (1,1))
        #nx.draw(graph, pos = layout, node_color = colormap, node_size = [100] * len(colormap), ax = ax)
        #nx.draw_networkx(graph, node_color = colormap, with_labels = True, ax = ax)
        nx.draw_networkx_labels(graph, pos=layout, labels = labels, font_size = 5, ax = ax)
        if draw_order:
            nx.draw_networkx_edge_labels(graph, pos=layout, edge_labels = edge_labels, ax = ax, font_size = 5)
        return
    else:
        #plt.margins(x = 0.01, y = 0.01)
        #plt.gca().set_xlim([1.05 * x for x in plt.gca().get_xlim()])
        #plt.gca().set_ylim([1.05 * y for y in plt.gca().get_ylim()])
        nx.draw(graph, pos = layout, node_color = colormap, node_size = [800] * len(colormap))
        nx.draw_networkx_labels(graph, pos=layout, labels = labels, font_size = 20)
        if draw_order:
            nx.draw_networkx_edge_labels(graph, pos=layout, edge_labels = edge_labels)

    if label:
        plt.text(0.5, 0.9, label, fontsize = 'xx-large', transform = plt.gca().transAxes)
    
    if return_pil:
        from PIL import Image
        import io
        im = io.BytesIO()
        plt.savefig(im, format = 'png')
        im = Image.open(im)
        plt.close()
        return im
    elif filename is None:
        plt.show()
    else:
        plt.savefig(filename, bbox_inches = "tight", pad_inches = 0.5)

    plt.close()

def draw_side_chain(graph, align = True):

    try:
        attachment_smiles = get_attachment_smiles(graph)
        img = draw_from_attachment_smiles(attachment_smiles, align)
    except Exception as e:
        import traceback
        #traceback.print_exc()
        #print(f"{e}: falling back to graph")
        img = draw_graph(graph, return_pil = True)

    return img

def get_attachment_smiles(graph):

    from macrocycles import write_smiles

    smiles = write_smiles.write_smiles(graph)

    #resolve over-bracketing
    smiles = smiles.replace("[N]", "N")
    smiles = smiles.replace("[S]", "S")
    smiles = smiles.replace("[O]", "O")
    smiles = smiles.replace("[C]","C")

    smiles = smiles.replace("*", "C")
    smiles = smiles.replace("&", "c")

    #handle backbone heteroatoms
    smiles = smiles.replace("[DX]", "C([*:2])(=O)")
    smiles = smiles.replace("[PX]", "N([*:1])")

    #handle backbone heteroatoms in cyclic side chains
    smiles = smiles.replace("[N|PX]", "N([*:1])")
    smiles = smiles.replace("[O|PX]", "O([*:1])")
    smiles = smiles.replace("[n|px]", "n([*:1])")
    smiles = smiles.replace("[o|px]", "o([*:1])")

    #resolve groups that don't have correct charges
    smiles = smiles.replace("N=N=N", "[N]=[N+]=[N-]")
    smiles = smiles.replace("N(O)=O", "[N+]([O-])=O")
    smiles = smiles.replace("N(=O)O", "[N+]([O-])=O")

    try:
        final_smiles = search_stereo_space(graph, smiles)
    except Exception as e:
        print(e)
        return None

    return final_smiles

def search_stereo_space(graph, smiles):

    import re
    
    pattern = re.compile("\?[R,S][H]?")

    options = []

    hits = []
    for hit in re.finditer(pattern, smiles):
        hits.append(hit)

    if len(hits) > 4:
        raise Exception("Too many stereocenters")

    import itertools
    from rdkit import Chem
    from rdkit.Chem.rdmolops import FindPotentialStereo
    s = ["@", "@@"]

    combs = list(itertools.product(s, repeat = len(hits)))

    for option in combs:
    
        fail = False
        last = 0
        build = ""
        for idx, hit in enumerate(hits):
            build += smiles[last:hit.start()]
            lc = smiles[hit.end() - 1]
            if lc == "H":
                hstr = "H"
            else:
                hstr = ""
            build += option[idx] + hstr + ":" + str(100 + (idx + 1) * 20)
            last = hit.end()

        build += smiles[last:]

        mol = Chem.MolFromSmiles(build)

        id_dict = {}
        for atom in mol.GetAtoms():
            if atom.HasProp("molAtomMapNumber"):
                if int(atom.GetProp("molAtomMapNumber")) > 100:
                    id_dict[atom.GetIdx()] = int(atom.GetProp("molAtomMapNumber"))

        for atom in mol.GetAtoms():
            atom.SetProp("molAtomMapNumber", "0")
        
    
        stereo_info = FindPotentialStereo(mol, cleanIt = True)

        d = {}
        for item in stereo_info:
            d[item.centeredOn] = str(item.descriptor)

        stripped = strip_map_number(build)

        mol = Chem.MolFromSmiles(stripped)

        for atom in mol.GetAtoms():

            atom_idx = atom.GetIdx()
            if atom_idx in id_dict:

                idx = int(((id_dict[atom_idx] - 100) / 20) - 1)
                hit = hits[idx]
                start = hit.start()
                end = hit.end()
                true_label = smiles[start:end][1]

                rdkit_assignment = d[atom_idx]

                cipcode = mol.GetAtomWithIdx(atom_idx).GetProp("_CIPCode")

                if true_label != cipcode:
                    fail = True

        if not fail:
            return stripped


    return None

def strip_map_number(s):

    ##only strips 3-digit map numbers

    import re
    pattern = re.compile("\:[0-9][0-9][0-9]]")
    return re.sub(pattern, "]", s)


def draw_from_attachment_smiles(attachment_smiles, align = True):

    from rdkit import Chem
    from rdkit.Chem import rdDepictor
    from rdkit.Chem.Draw.rdMolDraw2D import MolDrawOptions
    from rdkit.Chem.Draw import rdMolDraw2D
    from rdkit.Geometry.rdGeometry import Point2D, Point3D
    from rdkit.Chem.TemplateAlign import AlignMolToTemplate2D
    from rdkit.Chem import rdCIPLabeler
    from PIL import Image

    mol = Chem.MolFromSmiles(attachment_smiles)

    if mol is None:
        raise Exception("Nice drawing failed")

    from rdkit.Chem.rdmolops import FindPotentialStereo
    FindPotentialStereo(mol)

    alpha_template_smiles = "C(C(=O)*)N*"
    alpha_template = Chem.MolFromSmiles(alpha_template_smiles)

    beta_template_smiles = "C(C(=O)*)CN*"
    beta_template = Chem.MolFromSmiles(beta_template_smiles)

    gamma_template_smiles = "C(C(=O)*)CCN*"
    gamma_template = Chem.MolFromSmiles(gamma_template_smiles)

    rdDepictor.Compute2DCoords(alpha_template)
    rdDepictor.Compute2DCoords(beta_template)
    rdDepictor.Compute2DCoords(gamma_template)

    align = False

    if align:
        try:
            aligned_id = AlignMolToTemplate2D(mol,alpha_template,clearConfs=False)
        except Exception as e:
            print(e)
            try:
                aligned_id = AlignMolToTemplate2D(mol,beta_template,clearConfs=False)
            except:
                try:
                    aligned_id = AlignMolToTemplate2D(mol,gamma_template,clearConfs=False)
                except:
                    print("Warning: Could not align to a template")
                    rdDepictor.Compute2DCoords(mol)
                    aligned_id = -1
                    #raise Exception("Could not align to a template")
    else:
        rdDepictor.Compute2DCoords(mol)
        aligned_id = -1


    p_idx = None  # atom that represents the proximal connection point
    d_idx = None  # atom that represents the distal connection point
    pref = None  # atom that proximal connection point is connected to
    dref = None # atom that distal connection point is connected to
    
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == "*":
            number = atom.GetPropsAsDict()["molAtomMapNumber"]
            if number == 1:
                p_idx = atom.GetIdx()
            elif number == 2:
                atom.SetProp("_CIPRank", "0")
                d_idx = atom.GetIdx()
            else:
                raise Exception

    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)

    bond_dict = {}

    for bond in mol.GetBonds():
        if bond.GetBeginAtomIdx() == p_idx:
            pref = bond.GetEndAtomIdx()
        if bond.GetBeginAtomIdx() == d_idx:
            dref = bond.GetEndAtomIdx()
        if bond.GetEndAtomIdx() == p_idx:
            pref = bond.GetBeginAtomIdx()
        if bond.GetEndAtomIdx() == d_idx:
            dref = bond.GetBeginAtomIdx()
            
    drawer = rdMolDraw2D.MolDraw2DCairo(400,400)
    options = MolDrawOptions()

    for atom in mol.GetAtoms():
        d = atom.GetPropsAsDict()
        if "molAtomMapNumber" in d:
            #options.atomLabels[atom.GetIdx()] = atom.GetSymbol()
            atom.ClearProp("molAtomMapNumber")

    options.atomLabels[p_idx] = ""
    options.atomLabels[d_idx] = ""

    options.continuousHighlight = True
    options.addStereoAnnotation = True
    options.fillHighlights = True
    options.baseFontSize = 0.1

    drawer.SetDrawOptions(options)

    drawer.DrawMoleculeWithHighlights(mol, "", {}, {}, {}, {})

    if p_idx is not None:
        ppos = mol.GetConformer(aligned_id).GetAtomPosition(p_idx)
        prefpos = mol.GetConformer(aligned_id).GetAtomPosition(pref)
        ppos = Point2D(ppos)
        pref = Point2D(prefpos)
        diff = pref - ppos
        between = ppos + (diff / 2) 
        drawer.DrawAttachmentLine(ppos, between, color = (0,0,0), len = 1.0)

    if d_idx is not None:
        dpos = mol.GetConformer(aligned_id).GetAtomPosition(d_idx)
        drefpos = mol.GetConformer(aligned_id).GetAtomPosition(dref)
        #c = b + ((a - b) / 2)
        dpos = Point2D(dpos)
        dref = Point2D(drefpos)
        diff = dref - dpos
        between = dpos + (diff / 2) 
        drawer.DrawAttachmentLine(dpos, between, color = (0,0,0), len = 1.0)

    drawer.FinishDrawing()
    png_text = drawer.GetDrawingText()

    from io import BytesIO
    
    fobj = BytesIO(png_text)
    img = Image.open(fobj)
    return img
    
