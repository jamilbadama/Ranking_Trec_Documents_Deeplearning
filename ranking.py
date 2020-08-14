import csv
import numpy as np
import pandas as pd
import nltk.data
import pickle
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi, BM25L, BM25Plus


def load_csv_dataset():
    breast_cancer = pd.read_csv("data/dataset/1_Breast Cancer.csv", encoding='latin1')
    healthy = pd.read_csv("data/dataset/2_Healthy.csv", encoding='latin1')
    hiv = pd.read_csv("data/dataset/3_HIV.csv", encoding='latin1')
    melanoma = pd.read_csv("data/dataset/4_Melanoma.csv", encoding='latin1')
    prostate_cancer = pd.read_csv("data/dataset/5_Prostate Cancer.csv", encoding='latin1')

    queries = pd.read_csv("data/dataset/queries.csv", encoding='latin1')

    all_data = pd.concat([breast_cancer, healthy, hiv, melanoma, prostate_cancer], ignore_index=True)

    return (breast_cancer, healthy, hiv, melanoma, prostate_cancer, all_data, queries)


breast_cancer, healthy, hiv, melanoma, prostate_cancer,all_data, queries=load_csv_dataset()

embedder = SentenceTransformer('bert-base-nli-mean-tokens')

#Regenerate embedding code for each Disease category Change the file name and data according to disease

# corpus_embeddings = embedder.encode(prostate_cancer.Abstract.tolist())
# embedding_file = "data/models/prostate_cancer_embedding.emb"
# with open(embedding_file,mode='wb') as emb_f:
#     pickle.dump(corpus_embeddings,emb_f)


def query_ranking(topic, index, query, gene, topk=10):
    corpus = []
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    if topic == "HIV":
        Title = hiv['Title']
        Abstract = hiv['Abstract']
        NCT_ID = hiv['NCT_ID']

        for article in hiv.Abstract.apply(lambda row: row.lower()):
            corpus.extend(tokenizer.tokenize(article))

        embedding_file = "data/models/embeddings/trec-embedding/hiv_embedding.emb"
        with open(embedding_file, mode='rb') as emb_f:
            corpus_embeddings = pickle.load(emb_f)

    elif topic == "Breast Cancer":
        Title = breast_cancer['Title']
        Abstract = breast_cancer['Abstract']
        NCT_ID = breast_cancer['NCT_ID']

        for article in breast_cancer.Abstract.apply(lambda row: row.lower()):
            corpus.extend(tokenizer.tokenize(article))

        embedding_file = "data/models/embeddings/breastCander_embedding.emb"
        with open(embedding_file, mode='rb') as emb_f:
            corpus_embeddings = pickle.load(emb_f)


    elif topic == "Melanoma":
        Title = melanoma['Title']
        Abstract = melanoma['Abstract']
        NCT_ID = melanoma['NCT_ID']

        for article in melanoma.Abstract.apply(lambda row: row.lower()):
            corpus.extend(tokenizer.tokenize(article))

        embedding_file = "data/models/embeddings/melanoma_embedding.emb"
        with open(embedding_file, mode='rb') as emb_f:
            corpus_embeddings = pickle.load(emb_f)
    elif topic == "Healthy":
        Title = healthy['Title']
        Abstract = healthy['Abstract']
        NCT_ID = healthy['NCT_ID']

        for article in healthy.Abstract.apply(lambda row: row.lower()):
            corpus.extend(tokenizer.tokenize(article))

        embedding_file = "data/models/embeddings/healthy_embedding.emb"
        with open(embedding_file, mode='rb') as emb_f:
            corpus_embeddings = pickle.load(emb_f)

    elif topic == "Prostate Cancer":
        Title = prostate_cancer['Title']
        Abstract = prostate_cancer['Abstract']
        NCT_ID = prostate_cancer['NCT_ID']

        for article in prostate_cancer.Abstract.apply(lambda row: row.lower()):
            corpus.extend(tokenizer.tokenize(article))

        embedding_file = "data/models/embeddings/prostate_cancer_embedding.emb"
        with open(embedding_file, mode='rb') as emb_f:
            corpus_embeddings = pickle.load(emb_f)

    else:
        Title = all_data['Title']
        Abstract = all_data['Abstract']
        NCT_ID = all_data['NCT_ID']

        for article in all_data.Abstract.apply(lambda row: row.lower()):
            corpus.extend(tokenizer.tokenize(article))

        embedding_file = "data/models/embeddings/all_data_embedding.emb"
        with open(embedding_file, mode='rb') as emb_f:
            corpus_embeddings = pickle.load(emb_f)

    bm25 = BM25Okapi(corpus)
    tokenized_gene = gene.split(" ")
    BM25_Score = bm25.get_scores(tokenized_gene) * 2
    query_embeddings = embedder.encode(query)
    score_corpus = np.sum(query_embeddings * corpus_embeddings, axis=1) / np.linalg.norm(corpus_embeddings, axis=1)

    topk_idx = np.argsort(score_corpus)[::-1][:topk]
    i = 0
    for idx in topk_idx:
        i = i + 1
        score = score_corpus[idx] + BM25_Score[idx]
        print(index, '0', NCT_ID[idx], 1)
        with open('data/Ranking_CWE_BM25_results_1.csv', 'a', newline='') as csvfile:
            fieldnames = ['QueryNum', 'Q0', 'NCT_ID', 'Score', ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            cal_score = 0
            if score > 4.0 and score < 5.0:
                cal_score = 1
            elif score > 5.0 and score < 10.0:
                cal_score = 2
            else:
                cal_score = 0

            writer.writerow({'QueryNum': index, 'Q0': '0', 'NCT_ID': NCT_ID[idx], 'Score': score})

if __name__ == '__main__':
    for index, row in queries.iterrows():
        print(query_ranking("Melanoma", index + 1, row["Query"], row["Gene"], 1000))