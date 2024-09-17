# ASUMSI untuk tugas ini adalah hasil retrieval BM25 adalah ground truth
# Anda akan mengevaluasi hasil retrieval oleh TF-IDF, seberapa baik hasilnya

# Anda diberikan file queries.txt yang berisi nama query yang akan dijadikan sebagai data pengujian
# Lakukan retrieval BM25 top-20 untuk masing-masing query tersebut, lalu simpan output-nya
# ke dalam text file "qrels.txt"

from bsbi import BSBIIndex
from tqdm import tqdm
from compression import VBEPostings
import math

BSBI_instance = BSBIIndex(data_dir='arxiv_collections',
                          postings_encoding=VBEPostings,
                          output_dir='index')

def write_to_qrels(input_file='queries.txt', output_file='qrels.txt', k=20):
    """
    Fungsi ini untuk membuat data qrels, dengan asumsi bahwa hasil retrieval
    BM25 yang menjadi ground truth. Gunakan default parameter BM25 saja.

    Format output di text file:

    Parameters
    ----------
    input_file: str
        nama file input yang berisi queries
    output_file: str
        nama file untuk qrels
    k: int
        top-k results yang akan di-retrieve
    """
    queries = {}

    with open(input_file) as f:
        parsed_by_nl = str(f.read()).split("\n")
        for q in parsed_by_nl:
            q_splitted = q.split()
            queries[q_splitted[0]] = " ".join(q_splitted[1:])
    
    result = ""
    for q_id, query in tqdm(queries.items()):
        for (_, doc) in BSBI_instance.retrieve_bm25_taat(query, k):
            result += f"{q_id} {doc}\n"
    
    with open(output_file, "w") as f:
        f.write(result)
    
    print("Finish generating qrels!\n")

def read_from_qrels(input_file='qrels.txt'):
    """
    Fungsi ini membaca file qrels, lalu menyimpannya dalam sebuah dictionary.

    Parameters
    ----------
    input_file: str
        nama file input qrels
    
    Returns
    -------
    dict(str, List[str])
        key berupa query ID, value berupa list of document filenames
    """
    # TODO
    
    return dict()

def retrieve_and_generate_binary_relevancy_vector(q_docs_dict, queries_input='queries.txt', k=20):
    """
    Fungsi ini melakukan retrieval dengan TF-IDF, lalu hasilnya dibandingkan
    dengan ground truth di qrels.txt.

    Lakukan looping di semua dokumen hasil retrieval TF-IDF, lalu berikan nilai
    0 untuk dokumen yang ada di TF-IDF tapi tidak ada di qrels, dan berikan nilai
    1 untuk dokumen yang ada di keduanya.

    Misalnya:   ground truth = [D1, D3, D10, D12, D15]
                prediksi/retrieval = [D2, D1, D4, D10, D12]
                hasil = [0, 1, 0, 1, 1]
    
    Parameters
    ----------
    q_docs_dict: dict(str, List[str])
        dictionary dengan key berupa query ID dan value berupa list of relevant docs
    queries_input: str
        path ke file yang berisi mapping query id dan query
    k: int
        top-k result yang diinginkan
    
    Returns
    -------
    dict(str, List[int])
        key berupa query id, value berupa binary vector yang menunjukkan relevansi/ranking
    """
    # TODO

    return dict()

class Metrics:
    """
    Class yang berisi implementasi metrik-metrik yang akan diuji coba.

    Parameters
    ----------
    ranking: List[int]
        binary vector yang menunjukkan ranking
    """
    def __init__(self, ranking):
        self.ranking = ranking
    
    def rbp(self, p=0.8):
        """
        Rank-biased Precision (RBP)
        """
        # TODO
        return 0.
    
    def dcg(self):
        """
        Discounted Cumulative Gain (DCG)
        Gunakan log basis 2
        """
        # TODO
        return 0. 
    
    def ndcg(self):
        """
        Normalized DCG
        """
        # TODO
        return 0.

    def prec(self, k):
        """
        Precision@K
        """
        # TODO
        return 0.

    def ap(self):
        """
        Average Precision
        """
        # TODO
        return 0.

if __name__ == '__main__':
    write_to_qrels()
    q_docs_dict = read_from_qrels()
    q_ranking_dict = retrieve_and_generate_binary_relevancy_vector(q_docs_dict)

    eval = {
        "rbp": [],
        "dcg": [],
        "ndcg": [],
        "prec@5": [],
        "prec@10": [],
        "ap": []
    }    

    for (_, ranking) in q_ranking_dict.items():
        metrics = Metrics(ranking)
        eval['rbp'].append(metrics.rbp())
        eval['dcg'].append(metrics.dcg())
        eval['ndcg'].append(metrics.ndcg())
        eval['prec@5'].append(metrics.prec(5))
        eval['prec@10'].append(metrics.prec(10))
        eval['ap'].append(metrics.ap())

    # average of all queries
    for metric, scores in eval.items():
        print(f"Metrik {metric}: {sum(scores)/len(scores)}")
