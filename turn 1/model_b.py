import requests
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align.Applications import ClustalOmegaCommandline
from Bio import AlignIO
import pandas as pd
import numpy as np
from bokeh.plotting import figure, output_file, show
from bokeh.models import HoverTool

# Downloading genomic sequences from NCBI SRA
base_url = "https://www.ncbi.nlm.nih.gov/sra/srapi.cgi?rettype=fasta&retmode=text&cmd=GetFasta"
accessions = ["SRR5577121", "SRR5634515", "SRR5645178"]
sequences = {}
for accession in accessions:
    sra_url = f"{base_url}&accession={accession}"
    response = requests.get(sra_url)
    sequences[accession] = response.text
# Creating SeqRecord objects for each sequence
records = [SeqRecord(Seq(seq), id=acc) for acc, seq in sequences.items()]

# Performing multiple sequence alignment (MSA) using Clustal Omega
clustal_exe = "clustalo"  # Replace with the path to your Clustal Omega executable
aligner = ClustalOmegaCommandline(clustal_exe, infile="-", outfile="aligned_sequences.fasta")
stdout, stderr = aligner(inp="".join(record.format("fasta") for record in records))
alignment = AlignIO.read("aligned_sequences.fasta", "fasta")

from Bio.Phylo import calculate_conservation_scores

conservation_scores = calculate_conservation_scores(alignment)

from Bio.SeqUtils import GC_content, complexity

# Threshold for complexity
complexity_threshold = 0.4
potential_targets = []
for pos, score in enumerate(conservation_scores):
    if score > 0.5 and complexity(alignment[:, pos]) < complexity_threshold:
        potential_targets.append(pos)
