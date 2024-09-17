# BM-25 dengan konfigurasi parameter tertentu bisa diubah behavior-nya
# menjadi seperti TF-IDF

# Hyperparameter dari BM-25 yang bisa diubah-ubah adalah k1 dan b
# Silakan menentukan opsi k1 dan b Anda sendiri

from bsbi import BSBIIndex
from compression import VBEPostings

BSBI_instance = BSBIIndex(data_dir='arxiv_collections',
                          postings_encoding=VBEPostings,
                          output_dir='index')

# Isi dengan kandidat hyperparameter yang Anda inginkan
k1_candidates = []
b_candidates = []

query = "neural network"

# TODO: Lakukan hyperparameter tuning