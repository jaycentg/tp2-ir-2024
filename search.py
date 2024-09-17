from bsbi import BSBIIndex
from compression import VBEPostings

# sebelumnya sudah dilakukan indexing
# BSBIIndex hanya sebagai abstraksi untuk index tersebut
BSBI_instance = BSBIIndex(data_dir='arxiv_collections',
                          postings_encoding=VBEPostings,
                          output_dir='index')

query = input("Masukkan query Anda: ")

query_recc = [query]
query_recc += [query + subword for subword in BSBI_instance.get_query_recommendations(query)]

print("Rekomendasi query yang sesuai:")

for i in range(len(query_recc)):
    print(f"{i + 1}. {query_recc[i]}")

print()
choice = int(input("Masukkan nomor query yang Anda maksud: "))
chosen_query = query_recc[choice - 1]

print()
print(f"Pilihan Anda adalah '{chosen_query}'.")
print("Hasil pencarian:")

for (score, doc) in BSBI_instance.retrieve_bm25_taat(chosen_query, k=10):
    print(f"{doc} {score:>.3f}")