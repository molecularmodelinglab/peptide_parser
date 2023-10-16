from macrocycles.setup import ROOT_DIR
import os
import shutil
from copy import copy
import pandas as pd
from multiprocessing import Pool
from tqdm import tqdm
import logging
from macrocycles import utils
from macrocycles.common import STEREO
from macrocycles.common import RESIDUE_TYPE
from macrocycles.common import SIDE_CHAIN_TYPE
from macrocycles.common import BOND
from macrocycles.common import LD

def annotate_accessibility(molecule_ser, match_df, accessibility_dict, patg_dict, pcy1_dict):
    molecule_ser = molecule_ser.copy()
    match_df = match_df.copy()
    old_match_df = match_df.copy()

    annotated = []
    for i in range(len(match_df)):
        ser = match_df.iloc[i].copy()
        s = get_residue_accessibility(ser, accessibility_dict)
        ser = pd.concat((ser,s))
        annotated.append(ser)
    match_df = pd.DataFrame(annotated)
    match_df.index = old_match_df.index

    patg_info = get_patg_cyclizability(match_df, molecule_ser, patg_dict)
    molecule_ser = pd.concat((molecule_ser, patg_info))
    
    pcy1_info = get_pcy1_cyclizability(match_df, molecule_ser, pcy1_dict)
    molecule_ser = pd.concat((molecule_ser, pcy1_info))

    residues_noncanonical = match_df["Residue type"] == RESIDUE_TYPE.NONCANONICAL
    if molecule_ser["Simple class"] == "1":
        longest_run = utils.longest_run(residues_noncanonical, cyclic = True)
    else:
        longest_run = utils.longest_run(residues_noncanonical)

    molecule_ser["Longest NCAA run"] = longest_run
    molecule_ser["Number of NCAAs"] = sum(residues_noncanonical)

    residues_accessible = all(match_df["Residue accessible"])
    residues_accessible_ban_piz = all(match_df["Residue accessible ban piz"])

    residue_ban_reasons = []
    if molecule_ser["Simple class"] == "4":
        if any(match_df["Side chain type"] == SIDE_CHAIN_TYPE.CYCLIZATION):
            residue_ban_reasons.append("Thiopeptide has additional cyclization")

    residues_accessible = residues_accessible and len(residue_ban_reasons) == 0
    residues_accessible_ban_piz = residues_accessible_ban_piz and len(residue_ban_reasons) == 0

    clean_df = match_df.dropna(subset = ["Residue accessible"])
    num_inaccessible_residues = len(clean_df) - sum(clean_df["Residue accessible"])
    num_accessible_residues = len(clean_df) - num_inaccessible_residues
    num_ignored_residues = sum(pd.isnull(match_df["Residue accessible"]))

    molecule_ser["Residues accessible"] = residues_accessible
    molecule_ser["Residues accessible ban piz"] = residues_accessible_ban_piz
    molecule_ser["Residue accessibility ban reasons"] = residue_ban_reasons
    molecule_ser["Number of accessible residues"] = num_accessible_residues
    molecule_ser["Number of inaccessible residues"] = num_inaccessible_residues
    molecule_ser["Number of residues with ignored accessibility"] = num_ignored_residues

    accessibility_confidence = None
    if molecule_ser["Residues accessible"]:
        if molecule_ser["Number of NCAAs"] <= 2 and molecule_ser["Longest NCAA run"] <= 1:
            accessibility_confidence = "High"
        elif molecule_ser["Number of NCAAs"] <= 5 and molecule_ser["Longest NCAA run"] <= 3:
            accessibility_confidence = "Medium"
        elif molecule_ser["Number of NCAAs"] > 5 or molecule_ser["Longest NCAA run"] > 3:
            accessibility_confidence = "Low"
        else:
            print(molecule_ser["Longest NCAA run"])
            print(molecule_ser["Number of NCAAs"])
            raise Exception(f"Unhandled accessibility confidence for {molecule_ser['Molecule ID']}")

    molecule_ser["Residue accessibility confidence"] = accessibility_confidence

    return molecule_ser, match_df

def check_thioester_cyclizability(match_df, molecule_df):

    results = []
    for i in range(len(molecule_df)):
        ser = molecule_df.iloc[i]
        ban_reasons = []

        required_side_chains = ["Glycine", "Cysteine"]
        side_chains = list(ser["Side chain sequence"])
        cyclizable_side_chains = []
        for side_chain in side_chains:
            if side_chain in required_side_chains:
                cyclizable_side_chains.append(side_chain)

        if len(cyclizable_side_chains) == 0:
            ban_reasons.append("No cyclizable side chains")

        if side_chains.count("Cysteine") > 1:
            ban_reasons.append("Multiple cysteines")

        if side_chains.count("Lysine") > 0:
            ban_reasons.append("Lysine present")

        if side_chains.count("4ba0407b") > 0 or side_chains.count("e937d2f1"):
            ban_reasons.append("Banned UAA present")


        s = pd.Series()
        s["Thioester cyclization bans"] = ban_reasons
        s["Thioester cyclization accessible"] = len(ban_reasons) == 0
        s["Thioester cyclization almost accessible"] = len(ban_reasons) <= 1
        results.append(s)


    result_df = pd.DataFrame(results) 

    return result_df


def get_patg_cyclizability(match_df, molecule_ser, patg_dict):


    #continue ignoring n-methyl anything
    #switch to jarret's table

    patg_dict = patg_dict["Unmodified"]

    min_residues = 6
    max_residues = 11

    s = {}
    molecule_id = molecule_ser["Molecule ID"]

    ban_reasons = []
    cyclization_residues = []
    n_invalid_cyclization_residues = []
    d_invalid_cyclization_residues = []
    invalid_cyclization_residues = []

    if "peptide/macrocycle/1/" not in molecule_ser["Class"]:
        ban_reasons.append("not Class 1")


    if len(match_df) < min_residues:
        ban_reasons.append(f"Too few residues (<{min_residues})")
    if len(match_df) > max_residues:
        ban_reasons.append(f"Too many residues (>{max_residues})")

    for j in range(len(match_df)):
        match_ser = match_df.iloc[j]

        hash = match_ser["Everything hash"]

        cyclable = hash in patg_dict

        if cyclable:

            d_status = match_df["LD"].apply(lambda x: x == LD.D)
            lower = j - 1
            upper = (j + 1) % len(d_status)

            if d_status[lower] or d_status[upper]:
                d_invalid_cyclization_residues.append(j)
                continue
            
            side_chain_hashes = list(match_df["Everything hash"])

            #check direction here
            n_neighbor_index = (j + 1) % len(side_chain_hashes)

            neighbor_cyclable =  side_chain_hashes[n_neighbor_index] in patg_dict
            if neighbor_cyclable:
                invalid_cyclization_residues.append(j)
                continue

            cyclization_residues.append(j)

    if len(cyclization_residues) == 0 and len(d_invalid_cyclization_residues + n_invalid_cyclization_residues) == 0:
        ban_reasons.append("No cyclizable residues")
    if len(cyclization_residues) == 0 and len(n_invalid_cyclization_residues) > 0:
        ban_reasons.append("Cyclizable residues but N-neighbors are too")
    if len(cyclization_residues) == 0 and len(d_invalid_cyclization_residues) > 0:
        ban_reasons.append("Cyclizable residues but neighboring D")

    suffix = ""
    s["PatG ban reasons" + suffix] = ban_reasons
    s["PatG Cyclization accessible" + suffix] = (len(ban_reasons) == 0)
    s["PatG Cyclization almost accessible" + suffix] = (len(ban_reasons) <= 1)

    return pd.Series(s)

def get_pcy1_cyclizability(match_df, molecule_ser, pcy1_dict):

    pcy1_dict = pcy1_dict["Unmodified"]


    ban_reasons = []

    min_residues = 5
    max_residues = 9

    s = {}
    molecule_id = molecule_ser["Molecule ID"]
    tier = molecule_ser["Class"]
    ban_reasons = []
    cyclization_residues = []
    subdf = match_df[match_df["Molecule ID"] == molecule_id]

    if len(subdf) < min_residues:
        ban_reasons.append(f"Too few residues (<{min_residues})")
    if len(subdf) > max_residues:
        ban_reasons.append(f"Too many residues (>{max_residues})")

    d_invalid_cyclization_residues = []
    cyclable_status = []
    for j in range(len(subdf)):
        match_ser = subdf.iloc[j]
        #count total modifications or UAA's here?
        #check possible cyclization residues

        hash = match_ser["Everything hash"]
        
        cyclable = hash in pcy1_dict
        cyclable_status.append(cyclable)
        if cyclable:

            d_status = match_df["LD"].apply(lambda x: x == LD.D)
            lower = j - 1
            upper = (j + 1) % len(d_status)

            if d_status[lower] or d_status[upper]:
                #ban_reasons.append("Cyclizable residue, but neighboring D")
                d_invalid_cyclization_residues.append(j)
                continue
            
            cyclization_residues.append(j)

    if len(cyclization_residues) == 0 and len(d_invalid_cyclization_residues) == 0:
        ban_reasons.append("No cyclizable residues")
    if len(cyclization_residues) == 0 and len(d_invalid_cyclization_residues) > 0:
        ban_reasons.append("Cyclizable residues but neighboring D")

    s["PCY1 ban reasons"] = ban_reasons
    s["PCY1 cyclization accessible"] = (len(ban_reasons) == 0)
    s["PCY1 cyclization almost accessible"] = (len(ban_reasons) <= 1)

    return pd.Series(s)

def add_derivatives(names, derivative_df):

    import copy
    new_names = copy.copy(names)
    for name in names:
        idxs = derivative_df.iloc[:,derivative_df.columns == name].values
        import numpy as np
        hits = np.hstack(idxs == 1)
        hits = derivative_df.index[hits]
        new_names.extend(hits)

    return new_names

def output_spreadsheets(match_df, molecule_df, dirname, structure_dict, accessibility_filename = "../data/paper_accessibility.yaml"):
    import yaml

    accessibility_dict = yaml.safe_load(open(accessibility_filename))

    tiers = molecule_df["Simple class"].unique()


    cat_dict = {"D": "cat D",
                "N-mod": "cat N-methyl",
                "Beta": "cat Beta",
                "Ester": "cat Hydroxy Acid",
                "Unmodified": None
                }

    for category, d in accessibility_dict.items():

        category_column = cat_dict[category]
        print(category_column)

        names = d["Allowed"]
        print(len(names))

        for tier in tiers:
            filename = f"{dirname}/accessibility_rules/{category}/inaccessible/class_{tier}_{category}_inaccessible_side_chains.xlsx"
            print(filename)
            ids = molecule_df[molecule_df["Simple class"] == tier]["Molecule ID"]
            matches = match_df.loc[ids]
            if category_column is not None:
                matches = matches[matches[category_column]]

            matches = matches[matches["Side chain name"].apply(utils.shorten_name).apply(lambda x: x not in names)]
            if len(matches) == 0:
                continue
            counts = matches["Side chain name"].value_counts()
            counts = pd.DataFrame(counts)
            counts["Count"] = counts["Side chain name"]
            counts["Side chain name"] = counts.index
            utils.df_plus_side_chains(counts, filename, structure_dict)


            def get_full_name(short_name, structure_dict):

                if short_name[0].isupper():
                    return short_name

                shorts = set([utils.shorten_name(x) for x in structure_dict.keys()])
                assert(len(shorts) == len(structure_dict.keys()))
                short_dict = {utils.shorten_name(x):x for x in structure_dict.keys()}
                return short_dict[short_name]

        full_names = [get_full_name(x, structure_dict) for x in names]

        allowed_df = pd.DataFrame(full_names, columns = ["Side chain name"])
        filename = f"{dirname}/accessibility_rules/{category}/accessible/{category}_accessible_side_chains.xlsx"
        print(filename)
        utils.df_plus_side_chains(allowed_df, filename, structure_dict)


def read_accessibility_yaml(filename):

    import yaml
    accessibility_dict = yaml.safe_load(open(filename))

    new_dict = {}
    #enforce that multi-modification categories are alphabetical
    for key, val in accessibility_dict.items():
        if " and " in key:
            mods = key.split(" and ")
            mods.sort()
            new_key = " and ".join(mods)
            new_dict[new_key] = val
        else:
            new_dict[key] = val
    
    return new_dict

def get_residue_accessibility(ser, accessibility_dict):

    s = {}

    aa_name = ser["Side chain name"]
    hash = ser["Everything hash"]

    skipped_n_terminus = False
    ban_reasons = []
    modifications = []

    graph = ser["Everything graph"]
    if ser["Proximal bond"] == BOND.ESTER:
        modifications.append("Ester")
    if ser["cat N-methyl"]:
        modifications.append("N-mod")

    if ser["Side chain type"] == SIDE_CHAIN_TYPE.CYCLIZATION:
        pass
    elif utils.has_pyridine(graph):
        pass
    else:

        if ser["N-terminus"]:
            try:
                modifications.remove("N-mod")
                logging.info(f"Found n-terminus with n-methyl ({ser['Molecule ID']}), not checking for a ban here...")
            except:
                pass

        modifications = set(modifications)
        if modifications == set():
            key = "Unmodified"

        elif modifications == set(["Ester", "N-mod"]):
            key = "N-mod and Ester"

        elif modifications == set(["N-mod"]):
            key = "N-mod"

        elif modifications == set(["Ester"]):
            key = "Ester"
        else:
            raise Exception("Unhandled set of modifications for accessibility: {modifications}")

        if hash in accessibility_dict[key]:
            hits = accessibility_dict[key][hash]
        else:
            hits = []

        if len(hits) > 0:
            s["Residue accessible"] = True
        else:
            s["Residue accessible"] = False

        key = "Unmodified"

        if hash in accessibility_dict[key]:
            hits = accessibility_dict[key][hash]
        else:
            hits = []

        if len(hits) > 0:
            s["Side chain accessible"] = True
        else:
            s["Side chain accessible"] = False

    if "Residue accessible" in s:
        if "piperazate" in ser["Side chain name"].lower():
            s["Residue accessible ban piz"] = False
        else:
            s["Residue accessible ban piz"] = s["Residue accessible"]

    return pd.Series(s, dtype = 'object')

def parse_accessibility_database(filename, output_filename = None):

    from macrocycles import parsing
    from macrocycles import graphs
    from macrocycles import drawing

    from rdkit import Chem

    df = pd.read_csv(filename, sep = ',')
    smiles_col = "Smiles"
    help_smiles_col = "Manual attachment smiles"
    
    clean_df = df[pd.notnull(df[smiles_col])].copy()

    clean_df["skip parsing"] = clean_df["skip parsing"].apply(lambda x: False if pd.isnull(x) else bool(x)).astype(bool)

    clean_df = clean_df[~clean_df["skip parsing"]]


    clean_df["Is N-mod"] = clean_df["Is N-mod"].apply(lambda x: False if pd.isnull(x) else int(x)).astype(bool)

    clean_df["Is Ester"] = clean_df["Is Ester"].apply(lambda x: False if pd.isnull(x) else int(x)).astype(bool)

    clean_df = clean_df[clean_df["Method"].apply(lambda x: x in ["Canonical", "Fx", "Fx Chem", "Enzyme", "Fx Opt", "Fx EFP", "Natural aaRS", "Mutant aaRS", "NCA - EF-P"])]

    accessibility_dict = {"Unmodified": {}, "N-mod": {}, "Ester": {}, "N-mod and Ester": {}}

    name_to_hash = {}

    def yield_is_good(yield_string):

        if pd.isnull(yield_string):
            return True
        if yield_string == "x":
            return True
        elif "uM" in yield_string:
            s = yield_string.replace("uM", "")
            s = s.replace("<", "")
            s = float(s)
            #print(yield_string, s)
            return s > 0.06
        elif "%" in yield_string:
            s = float(yield_string.replace("%", ""))
            return s >= 10

        else:
            raise Exception

    parsing_failures = []
    yield_skips = 0

    for i in range(len(clean_df)):

        ser = clean_df.iloc[i]
        smiles = ser[smiles_col]

        yield_string = ser["yield annotation"]

        keep_yield = yield_is_good(yield_string)

        if not keep_yield:
            #print(f"skipping {ser['Name']} for poor yield {yield_string}")
            yield_skips += 1
            continue

        succeeded, info = parsing.get_info(smiles)

        if succeeded == False:
            #print(f"skipping {ser.name}, {ser['Name']} for parsing failure")
            #print(smiles)
            parsing_failures.append(smiles)
            continue

        hash = info.everything_hash
        name_to_hash[ser["Name"]] = hash


        if ser["Is N-mod"] and ser["Is Ester"]:
            key = "N-mod and Ester"
        elif ser["Is N-mod"]:
            key = "N-mod"
        elif ser["Is Ester"]:
            key = "Ester"
        else:
            key = "Unmodified"

        if hash not in accessibility_dict[key]:
            accessibility_dict[key][hash] = []
        accessibility_dict[key][hash].append(ser)
    

    if output_filename is not None:

        df_plus_images_with_attachment(clean_df, output_filename, smiles_col = smiles_col, attachment_col = help_smiles_col)

    if len(parsing_failures) > 0:
        logging.warning("Accessibility parsing failures: %d" % len(parsing_failures))
        logging.warning(parsing_failures)
    if yield_skips > 0:
        logging.warning("Accessibility yield skips: %d" % yield_skips)

    return accessibility_dict, name_to_hash

def df_plus_images_with_attachment(df, output_filename, smiles_col = None, attachment_col = None, graph_col = None):

    from macrocycles import parsing
    from macrocycles import graphs
    from rdkit import Chem
    from PIL import Image

    images = []

    for i in range(len(df)):

        if smiles_col is not None and graph_col is not None:
            raise Exception("Provided both graph col and smiles col to df drawing?")
        elif smiles_col is not None and graph_col is None:


            ser = df.iloc[i]
            attachment_smiles = ser[attachment_col]
            if pd.notnull(ser["align drawing"]) and ser["align drawing"] == False:
                align = False
            else:
                align = True
            if pd.notnull(attachment_smiles):
                try:
                    img = graphs.draw_from_attachment_smiles(attachment_smiles, align = align)
                except:
                    img = None
            else:
                smiles = ser[smiles_col]
                mol = Chem.MolFromSmiles(smiles)
                name = ser["Name"]
                try:
                    succeeded, info = parsing.get_info(smiles)
                    assert(succeeded == True)
                    img = graphs.draw_side_chain(info.everything_graph, align = align)
                except Exception as e:
                    print(e)
                    img = graphs.draw_graph(info.everything_graph, draw_order = True, use_stereo = True, return_pil = True)
            images.append(img)

        elif graph_col is not None and smiles_col is None:

            ser = df.iloc[i]
            graph = ser[graph_col]
            try: 
                img = graphs.draw_side_chain(graph, align = False)
            except Exception as e:
                print(e)
                img = graphs.draw_graph(graph, draw_order = True, use_stereo = True, return_pil = True)
            images.append(img)



    import xlsxwriter
    import os
    import io

    '''
    output_dirname = "/".join(output_filename.split("/")[:-1])
    os.makedirs(output_dirname, exist_ok = True)
    '''

    writer = pd.ExcelWriter(output_filename, engine = 'xlsxwriter')
    df.to_excel(writer, sheet_name = "Sheet1", index = True)
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']

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


    for i, img in enumerate(images):
        buf, _ = buffer_image(img)
        if buf != None:
            #worksheet.insert_image(i + 1, 3, "fake label", {'image_data': buf, 'object_position':3, 'x_scale':0.5, 'y_scale': 0.5})
            #worksheet.insert_image(i + 1, len(df.columns) + 1, "fake label", {'image_data': buf, 'object_position':3, 'x_scale':0.5, 'y_scale': 0.5})
            worksheet.insert_image(i + 1, len(df.columns) + 1, "fake label", {'image_data': buf, 'object_position':2, 'x_scale':0.5, 'y_scale': 0.5})
    writer.save()


def df_plus_images(df, output_filename, smiles_col = "Smiles"):

    from macrocycles import parsing
    from macrocycles import graphs
    from rdkit import Chem
    from PIL import Image

    images = []
    for i in range(len(df)):
        ser = df.iloc[i]
        smiles = ser[smiles_col]
        mol = Chem.MolFromSmiles(smiles)
        name = ser["Name"]
        try:
            info = parsing.get_info(smiles)
            img = graphs.draw_side_chain(info.with_stereo_graph)
        except:
            img = None
        images.append(img)

    import xlsxwriter
    import os
    import io

    '''
    output_dirname = "/".join(output_filename.split("/")[:-1])
    os.makedirs(output_dirname, exist_ok = True)
    '''

    writer = pd.ExcelWriter(output_filename, engine = 'xlsxwriter')
    df.to_excel(writer, sheet_name = "Sheet1", index = True)
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']

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


    for i, img in enumerate(images):
        buf, _ = buffer_image(img)
        if buf != None:
            #worksheet.insert_image(i + 1, 3, "fake label", {'image_data': buf, 'object_position':3, 'x_scale':0.5, 'y_scale': 0.5})
            #worksheet.insert_image(i + 1, len(df.columns) + 1, "fake label", {'image_data': buf, 'object_position':3, 'x_scale':0.5, 'y_scale': 0.5})
            worksheet.insert_image(i + 1, len(df.columns) + 1, "fake label", {'image_data': buf, 'object_position':2, 'x_scale':0.5, 'y_scale': 0.5})
    writer.save()




        






