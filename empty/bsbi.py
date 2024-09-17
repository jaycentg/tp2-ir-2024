import os
import pickle
import contextlib
import heapq
import math
import re
from porter2stemmer import Porter2Stemmer
import requests
import string

from index import InvertedIndexReader, InvertedIndexWriter
from trie import Trie
from util import IdMap, merge_and_sort_posts_and_tfs
from compression import VBEPostings
from tqdm import tqdm

class BSBIIndex:
    """
    Attributes
    ----------
    term_id_map(IdMap): Untuk mapping terms ke termIDs
    doc_id_map(IdMap): Untuk mapping relative paths dari dokumen (misal,
                    /collection/0/gamma.txt) to docIDs
    data_dir(str): Path ke data
    output_dir(str): Path ke output index files
    postings_encoding: Lihat di compression.py, kandidatnya adalah StandardPostings,
                    VBEPostings, dsb.
    index_name(str): Nama dari file yang berisi inverted index
    trie(Trie): Class Trie untuk query auto-completion
    """

    def __init__(self, data_dir, output_dir, postings_encoding, index_name="main_index"):
        self.term_id_map = IdMap()
        self.doc_id_map = IdMap()
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.index_name = index_name
        self.postings_encoding = postings_encoding
        self.trie = Trie()

        # Untuk menyimpan nama-nama file dari semua intermediate inverted index
        self.intermediate_indices = []

    def save(self):
        """Menyimpan doc_id_map, term_id_map, dan trie ke output directory via pickle"""

        with open(os.path.join(self.output_dir, 'terms.dict'), 'wb') as f:
            pickle.dump(self.term_id_map, f)
        with open(os.path.join(self.output_dir, 'docs.dict'), 'wb') as f:
            pickle.dump(self.doc_id_map, f)
        with open(os.path.join(self.output_dir, 'trie.pkl'), 'wb') as f:
            # file ini mungkin agak besar
            pickle.dump(self.trie, f)

    def load(self):
        """Memuat doc_id_map, term_id_map, dan trie dari output directory"""

        with open(os.path.join(self.output_dir, 'terms.dict'), 'rb') as f:
            self.term_id_map = pickle.load(f)
        with open(os.path.join(self.output_dir, 'docs.dict'), 'rb') as f:
            self.doc_id_map = pickle.load(f)
        with open(os.path.join(self.output_dir, 'trie.pkl'), 'rb') as f:
            self.trie = pickle.load(f)

    def parsing_block(self, block_path):
        """
        Lakukan parsing terhadap text file sehingga menjadi sequence of
        <termID, docID> pairs.

        Anda bisa menggunakan stemmer bahasa Inggris yang tersedia, seperti Porter Stemmer
        https://github.com/evandempsey/porter2-stemmer

        Untuk membuang stopwords, Anda dapat menggunakan library seperti NLTK.

        Untuk "sentence segmentation" dan "tokenization", bisa menggunakan
        regex atau boleh juga menggunakan tools lain yang berbasis machine
        learning.

        Parameters
        ----------
        block_path : str
            Relative Path ke directory yang mengandung text files untuk sebuah block.

            CATAT bahwa satu folder di collection dianggap merepresentasikan satu block.
            Konsep block di soal tugas ini berbeda dengan konsep block yang terkait
            dengan operating systems.

        Returns
        -------
        List[Tuple[Int, Int]]
            Returns all the td_pairs extracted from the block
            Mengembalikan semua pasangan <termID, docID> dari sebuah block (dalam hal
            ini sebuah sub-direktori di dalam folder collection)

        Harus menggunakan self.term_id_map dan self.doc_id_map untuk mendapatkan
        termIDs dan docIDs. Dua variable ini harus 'persist' untuk semua pemanggilan
        parsing_block(...).
        """
        # TODO
        # Hint: Anda dapat mengisi trie di sini

        return []

    def write_to_index(self, td_pairs, index):
        """
        Melakukan inversion td_pairs (list of <termID, docID> pairs) dan
        menyimpan mereka ke index. Disini diterapkan konsep BSBI dimana 
        hanya di-maintain satu dictionary besar untuk keseluruhan block.
        Namun dalam teknik penyimpanannya digunakan strategi dari SPIMI
        yaitu penggunaan struktur data hashtable (dalam Python bisa
        berupa Dictionary)

        ASUMSI: td_pairs CUKUP di memori

        Di Tugas Pemrograman 1, kita hanya menambahkan term dan
        juga list of sorted Doc IDs. Sekarang di Tugas Pemrograman 2,
        kita juga perlu tambahkan list of TF.

        Parameters
        ----------
        td_pairs: List[Tuple[Int, Int]]
            List of termID-docID pairs
        index: InvertedIndexWriter
            Inverted index pada disk (file) yang terkait dengan suatu "block"
        """
        # term_dict merupakan dictionary yang berisi dictionary yang
        # melakukan mapping dari doc_id ke tf
        term_dict = {}
        for term_id, doc_id in td_pairs:
            if term_id not in term_dict:
                term_dict[term_id] = dict()
            # Mengupdate juga TF (yang merupakan value dari dictionary yang di dalam)
            term_dict[term_id][doc_id] = term_dict[term_id].get(doc_id, 0) + 1
        
        for term_id in sorted(term_dict.keys()):
            # Sort postings list (dan tf list yang bersesuaian)
            sorted_postings_tf = dict(sorted(term_dict[term_id].items()))
            # Postings list adalah keys, TF list adalah values
            index.append(term_id, list(sorted_postings_tf.keys()), 
                         list(sorted_postings_tf.values()))

    def merge_index(self, indices, merged_index):
        """
        Lakukan merging ke semua intermediate inverted indices menjadi
        sebuah single index.

        Ini adalah bagian yang melakukan EXTERNAL MERGE SORT

        Gunakan fungsi merge_and_sort_posts_and_tfs(..) di modul util

        Parameters
        ----------
        indices: List[InvertedIndexReader]
            A list of intermediate InvertedIndexReader objects, masing-masing
            merepresentasikan sebuah intermediate inveted index yang iterable
            di sebuah block.

        merged_index: InvertedIndexWriter
            Instance InvertedIndexWriter object yang merupakan hasil merging dari
            semua intermediate InvertedIndexWriter objects.
        """
        merged_iter = heapq.merge(*indices, key=lambda x: x[0])
        curr, postings, tf_list = next(merged_iter)  # first item
        for t, postings_, tf_list_ in merged_iter:  # from the second item
            if t == curr:
                zip_p_tf = merge_and_sort_posts_and_tfs(list(zip(postings, tf_list)),
                                                        list(zip(postings_, tf_list_)))
                postings = [doc_id for (doc_id, _) in zip_p_tf]
                tf_list = [tf for (_, tf) in zip_p_tf]
            else:
                merged_index.append(curr, postings, tf_list)
                curr, postings, tf_list = t, postings_, tf_list_
        merged_index.append(curr, postings, tf_list)

    def _compute_score_tfidf(self, tf, df, N):
        """
        Fungsi ini melakukan komputasi skor TF-IDF.

        w(t, D) = (1 + log tf(t, D))       jika tf(t, D) > 0
                = 0                        jika sebaliknya

        w(t, Q) = IDF = log (N / df(t))

        score = w(t, Q) x w(t, D)
        Tidak perlu lakukan normalisasi pada score.

        Gunakan log basis 10.

        Parameters
        ----------
        tf: int
            Term frequency.

        df: int
            Document frequency.

        N: int
            Jumlah dokumen di corpus. 

        Returns
        -------
        float
            Skor hasil perhitungan TF-IDF.
        """
        # TODO
        return 0.
    
    def _compute_score_bm25(self, tf, df, N, k1, b, dl, avdl):
        """
        Fungsi ini melakukan komputasi skor BM25.
        Gunakan log basis 10 dan tidak perlu lakukan normalisasi.
        Silakan lihat penjelasan parameters di slide.

        Returns
        -------
        float
            Skor hasil perhitungan TF-IDF.
        """
        # TODO
        return 0.

    def retrieve_tfidf_daat(self, query, k=10):
        """
        Lakukan retrieval TF-IDF dengan skema DaaT.
        Method akan mengembalikan top-K retrieval results.

        Program tidak perlu paralel sepenuhnya. Untuk mengecek dan mengevaluasi
        dokumen yang di-point oleh pointer pada waktu tertentu dapat dilakukan
        secara sekuensial, i.e., seperti menggunakan for loop.

        Beberapa informasi penting: 
            1. informasi DF(t) ada di dictionary postings_dict pada merged index
            2. informasi TF(t, D) ada di tf_list
            3. informasi N bisa didapat dari doc_length pada merged index, len(doc_length)

        Parameters
        ----------
        query: str
            Query tokens yang dipisahkan oleh spasi

            contoh: Query "universitas indonesia depok" artinya ada
            tiga terms: universitas, indonesia, dan depok

        Result
        ------
        List[(int, str)]
            List of tuple: elemen pertama adalah score similarity, dan yang
            kedua adalah nama dokumen.
            Daftar Top-K dokumen terurut mengecil BERDASARKAN SKOR.

        JANGAN LEMPAR ERROR/EXCEPTION untuk terms yang TIDAK ADA di collection.

        """
        # TODO

        return []

    def retrieve_tfidf_taat(self, query, k=10):
        """
        Lakukan retrieval TF-IDF dengan skema TaaT.
        Method akan mengembalikan top-K retrieval results.

        Beberapa informasi penting: 
            1. informasi DF(t) ada di dictionary postings_dict pada merged index
            2. informasi TF(t, D) ada di tf_list
            3. informasi N bisa didapat dari doc_length pada merged index, len(doc_length)

        Parameters
        ----------
        query: str
            Query tokens yang dipisahkan oleh spasi

            contoh: Query "universitas indonesia depok" artinya ada
            tiga terms: universitas, indonesia, dan depok

        Result
        ------
        List[(int, str)]
            List of tuple: elemen pertama adalah score similarity, dan yang
            kedua adalah nama dokumen.
            Daftar Top-K dokumen terurut mengecil BERDASARKAN SKOR.

        JANGAN LEMPAR ERROR/EXCEPTION untuk terms yang TIDAK ADA di collection.

        """
        # TODO

        return []

    def retrieve_bm25_taat(self, query, k=10, k1=1.2, b=0.75):
        """
        Lakukan retrieval BM-25 dengan skema TaaT.
        Method akan mengembalikan top-K retrieval results.

        Parameters
        ----------
        query: str
            Query tokens yang dipisahkan oleh spasi

            contoh: Query "universitas indonesia depok" artinya ada
            tiga terms: universitas, indonesia, dan depok

        Result
        ------
        List[(int, str)]
            List of tuple: elemen pertama adalah score similarity, dan yang
            kedua adalah nama dokumen.
            Daftar Top-K dokumen terurut mengecil BERDASARKAN SKOR.
        """
        # TODO

        return []


    def do_indexing(self):
        """
        Base indexing code
        BAGIAN UTAMA untuk melakukan Indexing dengan skema BSBI (blocked-sort
        based indexing)

        Method ini scan terhadap semua data di collection, memanggil parsing_block
        untuk parsing dokumen dan memanggil write_to_index yang melakukan inversion
        di setiap block dan menyimpannya ke index yang baru.
        """
        # loop untuk setiap sub-directory di dalam folder collection (setiap block)
        for block_dir_relative in tqdm(sorted(next(os.walk(self.data_dir))[1])):
            td_pairs = self.parsing_block(block_dir_relative)
            index_id = 'intermediate_index_'+block_dir_relative
            self.intermediate_indices.append(index_id)
            with InvertedIndexWriter(index_id, self.postings_encoding, directory=self.output_dir) as index:
                self.write_to_index(td_pairs, index)
                td_pairs = None

        self.save()

        with InvertedIndexWriter(self.index_name, self.postings_encoding, directory=self.output_dir) as merged_index:
            with contextlib.ExitStack() as stack:
                indices = [stack.enter_context(InvertedIndexReader(index_id, self.postings_encoding, directory=self.output_dir))
                           for index_id in self.intermediate_indices]
                self.merge_index(indices, merged_index)

    def get_top_k_by_score(self, score_docs, k):
        """
        Method ini berfungsi untuk melakukan sorting terhadap dokumen berdasarkan score
        yang dihitung, lalu mengembalikan top-k dokumen tersebut dalam bentuk tuple
        (score, document). Silakan gunakan heap agar lebih efisien.

        Parameters
        ----------
        score_docs: Dictionary[int -> float]
            Dictionary yang berisi mapping docID ke score masing-masing dokumen tersebut.

        k: Int
            Jumlah dokumen yang ingin di-retrieve berdasarkan score-nya.

        Result
        -------
        List[(float, str)]
            List of tuple: elemen pertama adalah score similarity, dan yang
            kedua adalah nama dokumen.
            Daftar Top-K dokumen terurut mengecil BERDASARKAN SKOR.
        """
        # TODO

        return []
    
    def get_query_recommendations(self, query, k=5):
        # Method untuk mendapatkan rekomendasi untuk QAC
        # Tidak perlu mengubah ini
        self.load()
        last_token = query.split()[-1]
        recc = self.trie.get_recommendations(last_token, k)
        return recc

if __name__ == "__main__":

    BSBI_instance = BSBIIndex(data_dir='arxiv_collections',
                              postings_encoding=VBEPostings,
                              output_dir='index')
    BSBI_instance.do_indexing()  # memulai indexing!