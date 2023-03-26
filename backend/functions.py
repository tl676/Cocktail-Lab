from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


def make_matrix(data, binary=False, max_df=1.0, min_df=1, use_stop_words=True):
    """ Returns a #Doc by #Vocab term frequency matrix representation of data.

    By default this function returns a tf-idf matrix representation of data. 
    This can be switched to a binary representation by setting the binary param 
    to True.

    Parameters:
    data: str list
        The data to be vectorized. The list can be any length, as can the strings
        (i.e. a list of drinks' ingredients)
    binary: bool (Default = False)
        A flag to switch between tf-idf representation and binary representation
    max_df: float (Defualt = 1.0)
        The maximum document frequency to use for the matrix, as a proportion of 
        docs.
    min_df: float or int (Default = 1)
        The miniumum document frequency to use for the matrix. If [0.0,1.0], 
        the parameter represents a proportion of documents, otherwise in absolute
        doc counts. 
    use_stop_words: bool (Default = True)
        A flag to let sklearn remove common stop words.

    Returns:
    A #doc x #vocab np array

    """
    if binary:
        use_idf = False
        norm = None
    else:
        use_idf = True
        norm = 'l2'

    if use_stop_words:
        stop_words = 'english'
    else:
        stop_words = None

    tf_mat = TfidfVectorizer(max_df=max_df, min_df=min_df,
                             stop_words=stop_words, use_idf=use_idf,
                             binary=binary, norm=norm)

    return tf_mat.fit_transform([data]).toarray()


def make_query(tokens, doc_by_vocab):
    """ Returns a query vector made from tokens that matches the term matrix 
        doc_by_vocab.

    Parameters:
    tokens: str list or str set
        The tokens that make up a query. 
    doc_by_vocab: ?? by ?? np array
        The doc by vocab matrix as computed by make_matrix
    """
    vocab_to_index = {v: i for i, v in enumerate(
        doc_by_vocab.get_feature_names())}
    retval = np.zeros_like(doc_by_vocab[0])
    for t in tokens:
        try:
            ind = vocab_to_index[t]
            retval[ind] = 1
        except:
            # token not in matrix
            continue
    return retval


def cos_sim(vec1, vec2):
    """ Returns the cos sim of two vectors.

    Helper for cos_rank
    """
    num = np.dot(vec1, vec2)
    den = (np.linalg.norm(vec1)) * (np.linalg.norm(vec2))
    return num / den


def cos_rank(query, doc_by_vocab):
    """ Returns a tuple list that represents the doc indexes and their
        similarity scores to the query.

        Known Problems: This needs to be updated to use document ids (when we 
                        make those).

        Params:
        query: ?? by 1 string np array
            The desired ingredients, as computed by make_query
        doc_by_vocab: ?? by ?? np array
            The doc by vocab matrix as computed by make_matrix

        Returns:
            An (int, int) list where list[0] is the doc index, and list[1] is
            the similarity to the query.
    """
    retval = []

    for d in range(len(doc_by_vocab)):
        doc = doc_by_vocab[d]
        sim = cos_sim(query, doc)
        retval.append(d, sim)
    return list(sorted(retval, reverse=True, key=lambda x: x[1]))


def boolean_search(query, doc_by_vocab):
    """ Returns a list of doc indexes that contain all the query words

    Params:
    query: ?? by 1 string np array
        The desired ingredients, as computed by make_query
    doc_by_vocab: ?? by ?? np array
        The doc by vocab matrix as computed by make_matrix

    Returns:
        An int list where each element is the doc index of a doc containing 
        all the query words.
    """
    retval = []
    target = np.sum(query)
    for d in range(len(doc_by_vocab)):
        doc = doc_by_vocab[d]
        combine = np.logical_and(query, doc)
        if np.sum(combine) == target:
            retval.append(d)
    return retval
