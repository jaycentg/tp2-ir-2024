# Pada experiment ini, Anda akan coba membandingkan bagaimana performa retrieval 
# dengan menggunakan TaaT dan DaaT

# Kode ini tidak perlu diubah
# Cukup tambahkan WAND jika Anda mengimplementasikannya

from bsbi import BSBIIndex
from compression import VBEPostings
import time

BSBI_instance = BSBIIndex(data_dir='arxiv_collections',
                          postings_encoding=VBEPostings,
                          output_dir='index')

query = "neural network"

# retrieve top-500 supaya lebih keliatan perbedaannya
start = time.time()
BSBI_instance.retrieve_tfidf_daat(query, 500)
end = time.time()

print(f"Evaluasi TF-IDF dengan skema DaaT: {end - start} s")

start = time.time()
BSBI_instance.retrieve_tfidf_taat(query, 500)
end = time.time()

print(f"Evaluasi TF-IDF dengan skema TaaT: {end - start} s")

start = time.time()
BSBI_instance.retrieve_bm25_taat(query, 500)
end = time.time()

print(f"Evaluasi BM25 dengan skema TaaT: {end - start} s")