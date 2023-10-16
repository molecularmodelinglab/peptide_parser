import copy
import matplotlib.pyplot as plt
from matplotlib import colors
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdDepictor
import svgutils.transform as sg
from svgutils.transform import SVGFigure
from collections import defaultdict

from macrocycles import utils
from macrocycles import common

def highlight_accessibility(mol_ser, match_df, draw_labels = True):

    mol = mol_ser["ROMol"]

    labeled_matches = defaultdict(list)
    atom_sets  = []
    for i in range(len(match_df)):

        match_ser = match_df.iloc[i]

        ############# handle coloring
        bset = set(match_ser["Backbone atom indices"])
        sset = set(match_ser["Side chain atom indices"])

        if match_ser["Residue accessible"] == True:
            color = (0.5,1,0.5)
        elif match_ser["Residue accessible"] == False:
            color = (1,0.5,0.5)
        else:
            color = (0.5,0.5,0.5)
        atom_sets.append((sset, bset, color))
        ############# 


        ############# handle text
        label_text = ""
        name = match_ser["Side chain name"]
        if "Terminal cyclization" in match_ser and match_ser["Terminal cyclization"]:
                label_text += f"[T{match_ser['Terminal cyclization distance']}]"
        if "Internal cyclization" in match_ser and match_ser["Internal cyclization"]:
                label_text += f"[I{match_ser['Internal cyclization distance']}]"
        if all([not x.isdigit() for x in name]) or "-" in name:
            label_text += name
        else:
            label_text += name[:7]
        label_text += '   '

        if match_ser["Backbone type"] != common.BACKBONE_TYPE.ALPHA:
            if match_ser["Backbone type"] == common.BACKBONE_TYPE.BETA:
                label_text += "[β]"
            elif match_ser["Backbone type"] == common.BACKBONE_TYPE.GAMMA:
                label_text += "[γ]"
            else:
                raise Exception("Unknown backbone type in highlight_accessibility()")

        if match_ser['Proximal bond'] != common.BOND.PEPTIDE:
            if match_ser["Proximal bond"] == common.BOND.ESTER:
                label_text += "[EST]"
            else:
                raise Exception("Unhandled bond type")

        stereo = match_ser["Side chain carbon stereo"]
        stereo_str = ""
        if stereo == common.STEREO.S:
            stereo_str = "[S]"
        elif stereo == common.STEREO.R:
            stereo_str = "[R]"

        label_text += stereo_str

        if match_ser["Backbone nitrogen substituents"]:
            label_text += f"[Mod. N]"

        if label_text not in labeled_matches:
            labeled_matches[label_text] = []

        labeled_matches[label_text].append(color)
        ############

    atom_dict = defaultdict(list)
    bond_dict = defaultdict(list)

    for atom in mol.GetAtoms():
        for s in atom_sets:

            sset, bset, color = s
            if atom.GetIdx() in sset:
                atom_dict[atom.GetIdx()].append(color)

    for bond in mol.GetBonds():

        for s in atom_sets:

            sset, bset, color = s
            if bond.GetBeginAtomIdx() in sset and bond.GetEndAtomIdx() in sset:
                bond_dict[bond.GetIdx()].append(color)
            '''
            elif bond.GetBeginAtomIdx() in bset and bond.GetEndAtomIdx() in sset:
                bond_dict[bond.GetIdx()].append(color)
            elif bond.GetBeginAtomIdx() in sset and bond.GetEndAtomIdx() in bset:
                bond_dict[bond.GetIdx()].append(color)
            '''

    molSize = (700, 500)
    drawer = rdMolDraw2D.MolDraw2DSVG(molSize[0], molSize[1])
    opts = drawer.drawOptions()
    bond_width = 2
    opts.bondLineWidth = bond_width  # defaults to 2
    opts.highlightBondWidthMultiplier = 2

    drawer.DrawMoleculeWithHighlights(mol, "", dict(atom_dict), dict(bond_dict), {}, {})
    drawer.FinishDrawing()

    svg = drawer.GetDrawingText()
    svg = svg.replace('svg:', '')

    fig = sg.fromstring(svg)

    if draw_labels:
        buffer = 50 * len(match_df)
    else:
        buffer = 0

    label_fig = SVGFigure(height=molSize[1] + buffer , width=molSize[0])

    y = 20 + molSize[1]
    if draw_labels:
        id_text = sg.TextElement(50, y, match_ser["Molecule ID"], size=32,
                                       font='sans-serif', anchor='left', color="black")
        label_fig.append(id_text)
        y+=50

        for label_text, label_color in labeled_matches.items():
            label = sg.TextElement(80, y, label_text, size=32,
                                       font='sans-serif', anchor='left', color=colors.to_hex(label_color[0]))
            label_fig.append(label)
            y += 40

    label_fig.append(fig.getroot())
    svg_text = label_fig.to_str()

    return svg_text.decode("utf-8")

#match dict should be aa_name:list of list of atom ids, where each list is a pattern match
def moltosvg(mol,molSize=(700,500),kekulize=True, match_attributes = None, match_dict = None, label = None, label_indices = False, highlightatoms = None, backbone_atoms = None, filename = None, label_size = 6, for_notebook = False, id = "default_svg_id"):

    draw_labels = False



    mol = copy.copy(mol)
    if label_indices:
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(atom.GetIdx())

    used_colors = []
    labeled_matches = None
    if match_attributes is not None:
        labeled_matches = {}
        for i, match in match_attributes.iterrows():

            name, idx = i

            label_text = ""
            name = match["Side chain name"]
            if "Terminal cyclization" in match and match["Terminal cyclization"]:
                    label_text += f"[T{match['Terminal cyclization distance']}]"
            if "Internal cyclization" in match and match["Internal cyclization"]:
                    label_text += f"[I{match['Internal cyclization distance']}]"
            if all([not x.isdigit() for x in name]) or "-" in name:
                label_text += name
            else:
                label_text += name[:7]

            if match["Backbone type"] != common.BACKBONE_TYPE.ALPHA:
                label_text += f"[{match['Backbone type']}]"

            if match['Proximal bond'] != common.BOND.PEPTIDE:
                if match["Proximal bond"] == common.BOND.ESTER:
                    label_text += "[EST]"
                else:
                    raise Exception("Unhandled bond type")

            stereo = match["Side chain carbon stereo"]
            stereo_str = ""
            if stereo == common.STEREO.S:
                stereo_str = "[S]"
            elif stereo == common.STEREO.R:
                stereo_str = "[R]"

            label_text += stereo_str

            if match["Backbone nitrogen substituents"]:
                label_text += f"[Mod. N]"

            if label_text not in labeled_matches:
                labeled_matches[label_text] = []

            atom_to_number = list(set(match["Side chain atom indices"]).intersection(set(match["Backbone atom indices"])))[0]

            mol.GetAtomWithIdx(atom_to_number).SetProp("atomNote", str(idx))

            labeled_matches[label_text].append((set(match["Side chain atom indices"]),set(match["Backbone atom indices"])))
            used_colors.append(label_text)

    elif match_dict:
        labeled_matches = match_dict

    colormap = plt.get_cmap("rainbow")
    bond_dict = defaultdict(list)
    if labeled_matches:

        #set up reasonably separate colors for each key
        color_dict = {}
        names_for_colors = list(labeled_matches.keys())
        for i,name in enumerate(names_for_colors):
            relative_position = i / len(names_for_colors)
            #color = colormap(i / len(names_for_colors))
            color = colormap(0.3 + (relative_position * 0.7))
            color_dict[name] = tuple(color)

        atomdict = defaultdict(list)
        for aa_name, matches in labeled_matches.items():
            this_atoms = set()
            for match in matches:
                side_chain_indices, backbone_indices = match
                for atom in side_chain_indices:
                    atomdict[atom].append(color_dict[aa_name])
                    this_atoms.add(atom)

                for bond in mol.GetBonds():
                    if bond.GetBeginAtomIdx() in side_chain_indices and bond.GetEndAtomIdx() in side_chain_indices:
                        bond_dict[bond.GetIdx()].append(color_dict[aa_name])

    else:
        atomdict = None

    used_colors = [color_dict[x] for x in used_colors]

    if highlightatoms:
        colormap = plt.get_cmap("rainbow")
        atomdict = defaultdict(list)
        for atom in highlightatoms:
            atomdict[atom].append(tuple(colormap(0.5)))
        
    if backbone_atoms:
        backbone_dict = defaultdict(list)
        for bond in mol.GetBonds():
            if bond.GetBeginAtomIdx() in backbone_atoms and bond.GetEndAtomIdx() in backbone_atoms:
                backbone_dict[bond.GetIdx()].append((0.7,0.7,0.7))
        radius_dict = {}
        for key in backbone_dict.keys():
            radius_dict[key] = 1

    else:
        backbone_dict = None
        radius_dict = None



    if backbone_dict:
        a = backbone_dict
        b = bond_dict
        a.update(b)
        bond_dict = a

    mc = Chem.Mol(mol.ToBinary())

    for atom in mol.GetAtoms():
        if atom.HasProp("atomNote"):
            mc.GetAtomWithIdx(atom.GetIdx()).SetProp("atomNote", atom.GetProp("atomNote"))
    drawer = rdMolDraw2D.MolDraw2DSVG(molSize[0],molSize[1])

    opts = drawer.drawOptions()
    bond_width = 2
    opts.bondLineWidth = bond_width #defaults to 2
    #opts.scaleHighlightBondWidth = False
    opts.highlightBondWidthMultiplier = 2
    opts.annotationFontScale=1.5
    #rdDepictor.SetPreferCoordGen(True)
    if kekulize:
        try:
            Chem.Kekulize(mc)
        except:
            mc = Chem.Mol(mol.ToBinary())
    else:
        opts.prepareMolsBeforeDrawing=False
    if not mc.GetNumConformers():
        rdDepictor.Compute2DCoords(mc)


    if atomdict == None and backbone_dict == None:
        drawer.DrawMolecule(mc)
    elif atomdict == None:
        drawer.DrawMoleculeWithHighlights(mc, "", {}, dict(bond_dict), {}, {})
    elif backbone_dict == None:
        drawer.DrawMoleculeWithHighlights(mc, "", dict(atomdict), dict(bond_dict), {}, {})
        #drawer.DrawMoleculeWithHighlights(mc, "", dict(atomdict), {}, {}, {})
    else:
        drawer.DrawMoleculeWithHighlights(mc, "", dict(atomdict), dict(bond_dict), {}, {})
        #drawer.DrawMoleculeWithHighlights(mc, "", dict(atomdict), dict(backbone_dict), 1, 1)
    drawer.FinishDrawing()

    svg = drawer.GetDrawingText()
    svg = svg.replace('svg:','')

    fig = sg.fromstring(svg)

    if draw_labels:
        buffer = 50 * len(match_attributes)
    else:
        buffer = 0

    label_fig = SVGFigure(height = molSize[1] + buffer, width = molSize[0])
    label_fig.append(fig.getroot())
    #label_fig = copy.copy(fig)

    y = molSize[1] + 100
    if label:
        #label = sg.TextElement(int(molSize[0] / 2), molSize[1] - 10, label, size=label_size,
        #                       font='sans-serif', anchor='left', color="black")
        label = sg.TextElement(50, y, label, size=48,
                               font='sans-serif', anchor='left', color="black")
        label_fig.append(label)
        y += 40

    if labeled_matches != None and draw_labels:

        #y = 100
        for match_name, matches in labeled_matches.items():
            color = colors.to_hex(color_dict[match_name])
            '''
            if len(match_name) > 15:
                match_name = match_name[:5]
            label = sg.TextElement(molSize[0], y, f"{match_name}: {len(matches)}", size=32,
                               font='sans-serif', anchor='left', color=color)
            '''
            label = sg.TextElement(50, y, f"{match_name}: {len(matches)}", size=32,
                               font='sans-serif', anchor='left', color=color)
            label_bg = sg.TextElement(50, y, f"{match_name}: {len(matches)}", size=32,
                               font='sans-serif', anchor='left', color="black", weight = "bold")
            y += 40


            #label_fig.append(label_bg)
            label_fig.append(label)



    if filename:
        f = open(filename, 'wb')
        #f.write(fig.to_str())
        f.write(label_fig.to_str())
        f.close()

    fig = fig.to_str().decode("utf-8")
    label_fig = label_fig.to_str().decode("utf-8")

    return fig, label_fig, used_colors

def svg_post(s):

    out = ""

    for line in s.split("\n"):
        if "<rect" in line:
            continue
        if "</rect" in line:
            continue
        out += line

    out = out.replace("<svg", f'<svg class="svg_class"')
    return out

def peptide_report(molecule_ser, match_df):

    import pandas as pd
    retval = ""

    molecule_data_string = ""
    molecule_data_string = molecule_data_string[:-4]

    svg = highlight_accessibility(molecule_ser, match_df, draw_labels = False)

    _, labeled_svg, color_map = moltosvg(molecule_ser["ROMol"], match_attributes = match_df, backbone_atoms = molecule_ser["Backbone indices"])
    _, plain_svg, _ = moltosvg(molecule_ser["ROMol"], backbone_atoms = molecule_ser["Backbone indices"])

    plain_svg = svg_post(plain_svg)
    labeled_svg = svg_post(labeled_svg)
    svg = svg_post(svg)

    yes_color = "#92faa0"
    no_color = "#e85d5d"

    match_table_top = f'''
       <table class="styled-table">
      <thead>
      <tr>
        <th>Index</th>
        <th>Side chain name</th>
        <th>L/D</th>
        <th>Is Ester</th>
        <th>Is N-mod</th>
        <th>Residue accessible</th>
        <th>Drawing</th>
      </tr>
      </thead>
      <tbody>
    '''


    match_table_bottom = f'''
    </tbody>
    </table>
    '''

    match_table_middle = ""

    import os
    from macrocycles import graphs

    side_chain_page_dir = "side_chains/"
    if not os.path.exists(side_chain_page_dir):
        os.mkdir(side_chain_page_dir)

    for i in range(len(match_df)):

        s = match_df.iloc[i]
        color_tup = color_map[i]

        side_chain_dir = "side_chain_images"
        if not os.path.exists(side_chain_dir):
            os.mkdir(side_chain_dir)

        side_chain_page = side_chain_page_dir + clean_side_chain_name(s["Side chain name"]) + ".html"
        if not os.path.exists(side_chain_page):
            with open(side_chain_page, 'w') as f:
                f.write(get_side_chain_page(match_df.iloc[i]))



        side_chain_color = colors.to_hex(color_tup)

        side_chain_filename = f'{side_chain_dir}/{clean_side_chain_name(s["Side chain name"])}.png'


        if not os.path.exists(side_chain_filename):

            img = graphs.draw_side_chain(s["Everything graph"])
            img.save(side_chain_filename)

        name = s["Side chain name"]
        clean_name = clean_side_chain_name(name) 
        #name_with_link = f'<a href="side_chains/{clean_name}.html">{name}</a>'
        name_with_link = f'{name}'

        from macrocycles.common import BOND
        is_ester = s["Proximal bond"] == BOND.ESTER
        is_nmod = s["cat N-methyl"] or s["cat Non-methyl nitrogen modification"]
        ld = str(s['LD']).split(".")[-1]
        if ld == "OTHER":
            ld = ""

        print(s["Residue accessible"])
        is_residue_accessible_nan = pd.isna(s["Residue accessible"])
        is_residue_accessible = s["Residue accessible"] == True
        if is_residue_accessible_nan:
            color = "#888888"
            residue_accessible_text = "Not considered"
        else:
            if is_residue_accessible:
                color = yes_color
                residue_accessible_text = "True"
            else:
                color = no_color
                residue_accessible_text = "False"
        match_table_middle += f'''
        <tr>
        <td>{i}</td>
        <td style = "background-color: {side_chain_color}">{name_with_link}</td>
        <td>{ld}</td>
        <td>{is_ester}</td>
        <td>{is_nmod}</td>
        <td style = "background-color: {color}">{residue_accessible_text}</td>
        <td><div class="zoom"><img src="side_chain_images/{clean_name}.png" width="60" height="60"></div></td>
        </tr>
        '''

    match_table = match_table_top + match_table_middle + match_table_bottom

    prop_table_top = f'''
         <table class="styled-table">
      <thead>
      </thead>
      <tbody>
    '''

    prop_table_bottom = f'''
    </tbody>
    </table>
    '''

    keys = ["Molecule ID", "Simple class", "Class", "Automatic class", "Has undefined stereo", "Residue accessibility confidence", "Number of NCAAs"]
    access_keys = {"PatG Cyclization accessible":"PatG ban reasons", 
                   "PCY1 cyclization accessible":"PCY1 ban reasons",
                   "Residues accessible": "Residue accessibility ban reasons"}


    prop_table_middle = ""

    for key in keys:

        if key == "Has undefined stereo" and molecule_ser["Has undefined stereo"] is True:
            bg_color = no_color
            font_color = "white"
        elif key == "Class" and molecule_ser["Class"] != molecule_ser["Automatic class"]:
            bg_color = "yellow"
            font_color = "black"
        else:
            font_color = "black"
            bg_color = "white"

        prop_table_middle += "<tr>"
        prop_table_middle += f'<td style = "background-color: #aaaaaa" ><b><font color="white">{key}</font></b></td>'
        prop_table_middle += f'<td style = "background-color: {bg_color}" ><font color="{font_color}">{str(molecule_ser[key])}</font></td>'
        prop_table_middle += "</tr>"

    for access_key, ban_key in access_keys.items():
        val = molecule_ser[access_key]
        font_color = "black"
        if val is True:
            bg_color = yes_color
        else:
            bg_color = no_color

        prop_table_middle += "<tr>"
        prop_table_middle += f'<td style = "background-color: #aaaaaa" ><b><font color="white">{access_key}</font></b></td>'
        prop_table_middle += f'<td style = "background-color: {bg_color}" ><font color="{font_color}">{val}</font></td>'
        prop_table_middle += "</tr>"


        bans = molecule_ser[ban_key]
        ban_string = ",".join(bans)
        if val is False:
            bg_color = "white"
            prop_table_middle += "<tr>"
            prop_table_middle += f'<td style = "background-color: #aaaaaa" ><b><font color="white">{ban_key}</font></b></td>'
            prop_table_middle += f'<td style = "background-color: {bg_color}" ><font color="{font_color}">{ban_string}</font></td>'
            prop_table_middle += "</tr>"




    prop_table = prop_table_top + prop_table_middle + prop_table_bottom

    retval = f'''
    <body>
     <div class="row">
      <div class="column">

    {match_table}
    <br>
    {prop_table}
    </div>


      <div class="column">
    <!-- <center><h2>Highlighted side chains:</h2></center><br> -->
        {labeled_svg}
        <br>
        {svg}


      </div>
    </div>
    </body>
    </html>
    '''


    retval = f'''
    <body>
    <div class="panes">
    <div class="within_pane">

    <div class = "item">
    <center>
    <h3> Backbone </h3>
    </center>
    {plain_svg}
    <center>
    </center>
    </div>

    <div class = "item">
    <center>
    <h3> Side chains </h3>
    </center>
    {labeled_svg}
    </div>

    <div class = "item">
    <center>
    <h3> Residue accessibility </h3>
    </center>
    {svg}
    </div>

    </div>

    <div class="within_pane">

    <div class = "item">
    {match_table}
    </div>

    <div class = "item">
    {prop_table}
    </div>

    </div>

    '''

    retval += '''
    <script>
    //var svgEl = document.getElementById("default_svg_id");
    var svgEls = document.getElementsByClassName("svg_class");

    for (let i = 0; i< svgEls.length; i++) {
        let svgEl = svgEls[i];
        console.log(svgEl);
        const bbox = svgEl.getBBox();
        console.log(bbox);
        svgEl.setAttribute("viewBox", `${bbox.x} ${bbox.y} ${bbox.width} ${bbox.height}`);
        svgEl.setAttribute("width", `${bbox.width}`);
        svgEl.setAttribute("height", `${bbox.height}`);
        console.log(svgEl);

            }
    </script>
    '''

    return retval
    

def non_peptide_report(molecule_ser, match_df):

    _,molsvg,_ = moltosvg(molecule_ser["ROMol"])

    s = f'''
    <br>
    <br>
    <center>
    {molsvg}
    </center>
    <br>
    <center>
    This molecule is likely not a peptide.
    </center>
    '''
    return s

def clean_side_chain_name(s):

    return s.replace("_", "_").replace("/", "_").replace("#", "")



def html_report(molecule_ser, match_df, filename = None):


    if molecule_ser["Class"].split("/")[0] != "peptide":
        mol_report = non_peptide_report(molecule_ser, match_df)
    else:
        mol_report = peptide_report(molecule_ser, match_df)

    header = '''
        <html>
        <head>
        ''' + f'''
        <title>{molecule_ser["Molecule ID"]}</title>
        ''' + '''
        <style>

        html * {
          font-size: 20px;
          line-height: 1.625;
          color: #2020131;
          font-family: Arial, sans-serif;
        }
        .zoom {
          transition: transform .2s; /* Animation */
          width: 60px;
          height: 60px;
          margin: 0 auto;
        }

        .zoom img {
                width = 100%;
                transition = 0.5s all ease-in-out;
                }

        .zoom:hover img{
          transform: scale(5);
        }

        .column {
          float: left;
          margin: 30px;
        }

        .left {
          width: 50%;
        }

        .right {
          width: 50%;
        }

        /* Clear floats after the columns */
        .row:after {
          content: "";
          display: table;
          clear: both;
        }
        .styled-table {
        border-collapse: collapse;
        margin: 25px 0;
        font-size: 0.9em;
        font-family: sans-serif;
        min-width: 600px;
        box-shadow: 0 0 20px rgba(0, 0, 0, 0.15);
        }

        .styled-table thead tr {
        background-color: #aaaaaa;
        color: #ffffff;
        text-align: center;
        }

        .styled-table th,
        .styled-table td {
        padding: 12px 15px;
        }

        .styled-table tbody tr {
        border-bottom: 1px solid #333333;
        text-align: center;
        }

        .panes {
            display: flex;
            flex-flow: column wrap;
            justify-content:center;
        }

        .within_pane {
                display: flex;
                flex_flow: row wrap;
            justify-content: center;

                }

        .item {
                flex_shrink: 3;
                margin: 20px;
                border: 5px;
                }

        .topbar {
            background-color: #eeeeee;
            font-size: 40px;
            text-align: center;

         }


        }

        }

        }
        </style>

        </head>
        '''

    title_bar = f'<div class = "topbar">{molecule_ser["Molecule ID"]}</div>'

    retval = header + title_bar + mol_report

    if filename:

        with open(filename, 'w') as f:
            f.write(retval)

    return retval


def get_side_chain_page(ser):

    return "ayy"
