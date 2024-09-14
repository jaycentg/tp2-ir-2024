# BM-25 dengan konfigurasi parameter tertentu bisa diubah behavior-nya
# menjadi seperti TF-IDF

# Hyperparameter dari BM-25 yang bisa diubah-ubah adalah k1 dan b
# Silakan menentukan opsi k1 dan b Anda sendiri

from bsbi import BSBIIndex
from compression import VBEPostings

BSBI_instance = BSBIIndex(data_dir='arxiv_collections',
                          postings_encoding=VBEPostings,
                          output_dir='index')

query = "neural network"

print("BM25")
for (score, doc) in BSBI_instance.retrieve_bm25_taat(query, k1=1000000, b=0):
    print(doc, score)

print("===============================================")

print("TF-IDF")
for (score, doc) in BSBI_instance.retrieve_tfidf_taat(query):
    print(doc, score)