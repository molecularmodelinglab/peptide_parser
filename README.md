# peptide_parser

Python code for parsing molecular structures to identify peptides and detail their constituent residues. Focused on logic for detecting macrocyclization and keeping track of side chain counts across large datasets. Also contains a database of residues shown to be expressible in peptides along with basic heuristics for whether a peptide can be cyclized using either the PatG or PCY1 proteins.

## Setup

The code depends on a large number of libraries, particularly RDKit and NetworkX. We provide a conda environment file to install all of the necessary dependencies.

## Usage

Currently not super user friendly. `parsing.py` contains the `parse_mol()` function which should work for most purposes. Parsing results are returned internally and also can be summarized in an HTML file with drawings, highlighting, side chain information, and accessibility information.

## Source data

The manuscript uses the Supernatural, Lotus, and MiBig databases as a large pool of natural product molecules. Rather than handle license/sharing concerns for hosting each database here, we instead point users to the original sources and provide utilities for merging them:

Supernatural: https://bioinf-applied.charite.de/supernatural_3/index.php 

Lotus: https://lotus.naturalproducts.net/

MiBig: https://mibig.secondarymetabolites.org/

## Limitations

Natural peptides are complicated, and there will probably always be exceptions and edge cases and unhandled motifs in this code. We make an effort to test and validate on backbone cyclized, side-chain-to-backbone cyclized, side-chain-to-side-chain cyclized, and multi-cycle molecules. The provided test code has a few example moleucles from each of these classes. However, we know of some yet-unhandled cases:

- Discontiguous peptide backbones: If a molecule has multiple peptidic sections, the current algorithm will only choose the longest/best scored. This includes a large portion of thiopeptides, as an example
- High order amino acids: We only parse up to gamma amino acids, so anything delta and above will not parse correctly

It is almost certain that there are more cases that we don't cover. We encourage users to open discussions on improvements while maintaining accuracy on the current set of test cases.

## Citation

If you find this code to be useful, consider citing our manuscript that motivated it: ...
