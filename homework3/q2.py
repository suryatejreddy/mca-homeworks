import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def relevance_feedback(vec_docs, vec_queries, sim, n=10):
    """
    relevance feedback
    Parameters
        ----------
        vec_docs: sparse array,
            tfidf vectors for documents. Each row corresponds to a document.
        vec_queries: sparse array,
            tfidf vectors for queries. Each row corresponds to a document.
        sim: numpy array,
            matrix of similarities scores between documents (rows) and queries (columns)
        n: integer
            number of documents to assume relevant/non relevant

    Returns
    -------
    rf_sim : numpy array
        matrix of similarities scores between documents (rows) and updated queries (columns)
    """
    rf_sim = sim # change
    vec_queries = vec_queries.todense()
    vec_docs = vec_docs.todense()
    alpha = 0.7
    beta = 0.3

    for i in range(3):
        for q in range(len(vec_queries)):
            #Moving Closer To Related Documents
            related_documents_index  = np.argsort(-rf_sim[:, q])[:n] 
            related_documents = vec_docs[related_documents_index]
            to_add = alpha * np.sum(related_documents, axis = 0) / n

            #Moving Away From Unrelated Documents
            unrelated_documents_index = np.argsort(rf_sim[:, q])[:n]
            unrelated_documents = vec_docs[unrelated_documents_index]
            to_sub = -1 * beta * np.sum(unrelated_documents, axis = 0) / n

            vec_queries[q] += to_add + to_sub

        rf_sim = cosine_similarity(vec_docs, vec_queries)

    return rf_sim


def relevance_feedback_exp(vec_docs, vec_queries, sim, tfidf_model, n=10):
    """
    relevance feedback with expanded queries
    Parameters
        ----------
        vec_docs: sparse array,
            tfidf vectors for documents. Each row corresponds to a document.
        vec_queries: sparse array,
            tfidf vectors for queries. Each row corresponds to a document.
        sim: numpy array,
            matrix of similarities scores between documents (rows) and queries (columns)
        tfidf_model: TfidfVectorizer,
            tf_idf pretrained model
        n: integer
            number of documents to assume relevant/non relevant

    Returns
    -------
    rf_sim : numpy array
        matrix of similarities scores between documents (rows) and updated queries (columns)
    """

    rf_sim = sim # change
    vec_queries = vec_queries.todense()
    vec_docs = vec_docs.todense()
    vocab = vec_docs.T.dot(vec_docs)
    alpha = 0.4
    beta = 0.6


    for i in range(5):
        for q in range(len(vec_queries)):
            cur_query = vec_queries[q]

            most_important_word_index = np.argmax(cur_query)
            most_important_word_value = np.max(cur_query)
            most_similar_words_index = (-vocab[most_important_word_index]).argsort()[:, : 1]
            cur_query[:, most_similar_words_index] = most_important_word_value

            vec_queries[q] = cur_query

            #Moving Closer To Related Documents
            related_documents_index  = np.argsort(-rf_sim[:, q])[:n] 
            related_documents = vec_docs[related_documents_index]
            to_add = alpha * np.sum(related_documents, axis = 0) / n

            #Moving Away From Unrelated Documents
            unrelated_documents_index = np.argsort(rf_sim[:, q])[:n]
            unrelated_documents = vec_docs[unrelated_documents_index]
            to_sub = -1 * beta * np.sum(unrelated_documents, axis = 0) / n

            vec_queries[q] +=  to_add + to_sub
    
        rf_sim = cosine_similarity(vec_docs, vec_queries)
        
    return rf_sim