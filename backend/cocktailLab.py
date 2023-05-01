from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import csv
import math


class CocktailLab:
    def __init__(self):
        """Dictionary of {drink name: ingredients}"""
        self.cocktail_names_to_ingreds = self.read_file_ingreds(
            'data/final_cocktail_flavors_ingreds_popularity.csv')

        self.cocktail_names_to_ingreds_only = self.read_file_ingreds(
            'data/final_cocktail_flavors_ingreds_popularity.csv', ingreds_only=True)

        """Dictionary of {drink name: flavors}"""
        self.cocktail_names_to_flavors = self.read_file_flavors(
            'data/final_cocktail_flavors_ingreds_popularity.csv')

        """Dictionary of {drink name: popularity}"""
        self.cocktail_names_to_popularity = self.read_file_popularity(
            'data/final_cocktail_flavors_ingreds_popularity.csv')

        """Number of cocktails"""
        self.num_cocktails = len(self.cocktail_names_to_ingreds)

        """Dictionary of {cocktail name: index}"""
        self.cocktail_name_to_index = {
            name: index for index, name in
            enumerate(self.cocktail_names_to_ingreds.keys())
        }

        """Dictionary of index: cocktail name"""
        self.cocktail_index_to_name = {
            v: k for k, v in self.cocktail_name_to_index.items()}

        """List of cocktail names"""
        self.cocktail_names = self.cocktail_names_to_ingreds.keys()

        """The sklearn TfidfVectorizer object"""
        self.ingreds_tfidf_vectorizer = self.make_vectorizer(binary=True)

        self.ingreds = [self.cocktail_names_to_ingreds[cocktail] for cocktail in
                        self.cocktail_names_to_ingreds]

        """The term-document matrix"""
        self.ingreds_doc_by_vocab = self.ingreds_tfidf_vectorizer.fit_transform(
            self.ingreds).toarray()

        """Dictionary of {index: token}"""
        self.index_to_vocab = {i: v for i, v in enumerate(
            self.ingreds_tfidf_vectorizer.get_feature_names())}

        self.rocchio_alpha = 1.0

        self.rocchio_beta = 1.0

    def read_file_ingreds(self, file, ingreds_only=False):
        """ Returns a dictionary of format {'cocktail name' : 'ingred1,ingred2'}
        Parameters:
        file: name of file
        """
        with open(file, encoding="utf8") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            out = {}
            for row in csv_reader:
                if line_count == 0:
                    line_count += 1
                else:
                    if ingreds_only:
                        out[row[1].lower()] = row[3].lower()
                    else:
                        out[row[1].lower()] = row[5].lower()
        return out

    def read_file_flavors(self, file):
        """ Returns a dictionary of format {'cocktail name' : 'flavor1,flavor2'}
        Parameters:
        file: name of file
        """
        with open(file, encoding="utf8") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            out = {}
            for row in csv_reader:
                if line_count == 0:
                    line_count += 1
                else:
                    out[row[1].lower()] = ', '.join(row[6].lower().split(","))
        return out

    def read_file_popularity(self, file):
        """ Returns a dictionary of format {'cocktail name' : 'flavor1,flavor2'}
        Parameters:
        file: name of file

        """
        with open(file, encoding="utf8") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            out = {}
            for row in csv_reader:
                if line_count == 0:
                    line_count += 1
                else:
                    out[row[1].lower()] = float(row[7])
        return out

    def make_vectorizer(self, binary=False, max_df=1.0, min_df=1, use_stop_words=True):
        """ Returns a TfidfVectorizer object with the above preprocessing properties.

        By default this function returns a tf-idf matrix vectorizer. 
        This can be switched to a binary representation by setting the binary param 
        to True.

        Parameters:
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
        A #doc x #vocab np array vectorizer

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
                                 binary=binary, norm=norm,
                                 #  analyzer='word', token_pattern='[^,]+'
                                 )

        return tf_mat

    def make_query(self, tokens, vectorizer, doc_by_vocab):
        """ Returns a query vector made from tokens that matches the term matrix 
            doc_by_vocab.

        Parameters:
        tokens: str list or str set
            The tokens that make up a query. 
        vectorizer: tfidf vectorizer object
            The doc by vocab matrix as computed by make_matrix
        doc_by_vocab: tfidf or boolean matrix
        """
        vocab_to_index = {v: i for i, v in enumerate(
            vectorizer.get_feature_names())}
        retval = np.zeros_like(doc_by_vocab[0])
        for t in tokens:
            try:
                ind = vocab_to_index[t]
                retval[ind] = 1
            except:
                # token not in matrix
                continue
        return retval

    def cos_sim(self, vec1, vec2):
        """ Returns the cos sim of two vectors.

        Helper for cos_rank
        """
        num = np.dot(vec1, vec2)
        den = (np.linalg.norm(vec1)) * (np.linalg.norm(vec2))
        # TODO possibly throwing errors due to divide by 0
        return num / den

    def cos_rank(self, query, doc_by_vocab):
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
            sim = self.cos_sim(query, doc)
            # adjust by popularity
            sim += self.cocktail_names_to_popularity[self.cocktail_index_to_name[d]] * 0.02
            retval.append([d, sim])
        sorted_list = list(sorted(retval, reverse=True, key=lambda x: x[1]))
        return sorted_list

    def boolean_search_and(self, query, doc_by_vocab):
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
        for idx, doc in enumerate(doc_by_vocab):
            combine = np.logical_and(query, doc)
            if np.sum(combine) == target:
                retval.append(idx)
        return retval

    def boolean_search_not(self, query, doc_by_vocab):
        """ Returns a list of doc indexes that contain none of the query words

        Params:
        query: ?? by 1 string np array
            The desired ingredients, as computed by make_query
        doc_by_vocab: ?? by ?? np array
            The doc by vocab matrix as computed by make_matrix

        Returns:
            An int list where each element is the doc index of a doc containing 
            none of the query words.
        """
        retval = []
        for idx, doc in enumerate(doc_by_vocab):
            combine = np.logical_and(query, doc)
            if np.sum(combine) == 0:
                retval.append(idx)
        return retval

    def comma_space_split(self, str):
        return [i for i in ",".join(str.split(" ")).split(",") if i]

    def query(self, ingred_prefs=None, ingred_antiprefs=None, ingred_include=None, ingred_exclude=None,
              rocchio_plus=None, rocchio_minus=None):
        print(f"""prefs:{ingred_prefs}
        antiprefs:{ingred_antiprefs}
        include:{ingred_include}
        exclude:{ingred_exclude}
        rocchio:{rocchio_plus}, {rocchio_minus}""")

        # initialize variables
        matrix = self.ingreds_doc_by_vocab
        # [{'name': "cocktail name", 'flavors': 'cocktail flavors'}]
        rank_list = None
        # the list of indices to return (used by boolean and/not)
        idx_list = None
        # initialize as vector of 0s:
        flavor_prefs_vec = self.make_query(
            [""], self.ingreds_tfidf_vectorizer, matrix)
        # initialize as vector of 0s:
        flavor_antiprefs_vec = self.make_query(
            [""], self.ingreds_tfidf_vectorizer, matrix)
        flavor_include_vec = np.zeros(len(self.index_to_vocab))
        flavor_exclude_vec = np.zeros(len(self.index_to_vocab))
        rocchio_plus_vec = np.zeros(len(self.index_to_vocab))
        rocchio_minus_vec = np.zeros(len(self.index_to_vocab))
        cos_rank = None

        # vectorize inputs, if necessry
        if ingred_prefs:
            flavor_prefs_vec = self.make_query(
                [word.strip().lower()
                 for word in self.comma_space_split(ingred_prefs)],
                self.ingreds_tfidf_vectorizer,
                matrix)

        if ingred_antiprefs:
            # set the antipref flavors as -1
            flavor_antiprefs_vec = -1 * self.make_query(
                [word.strip().lower()
                 for word in self.comma_space_split(ingred_antiprefs)],
                self.ingreds_tfidf_vectorizer,
                matrix)

        if ingred_include:
            flavor_include_vec = self.make_query(
                [word.strip().lower()
                 for word in self.comma_space_split(ingred_include)],
                self.ingreds_tfidf_vectorizer,
                matrix)

        if ingred_exclude:
            flavor_exclude_vec = self.make_query(
                [word.strip().lower()
                 for word in self.comma_space_split(ingred_exclude)],
                self.ingreds_tfidf_vectorizer,
                matrix)

        if rocchio_plus:
            rocchio_plus_vec = self.ingreds_doc_by_vocab[int(
                rocchio_plus)] * self.rocchio_alpha

        if rocchio_minus:
            rocchio_minus_vec = - \
                self.ingreds_doc_by_vocab[int(
                    rocchio_minus)] * self.rocchio_beta

        # cosine sim:
        cos_rank = self.cos_rank(
            flavor_prefs_vec + flavor_antiprefs_vec + rocchio_plus_vec + rocchio_minus_vec, matrix)
        rank_list = [{
            'name': self.cocktail_index_to_name[i[0]],
            'ingredients': self.cocktail_names_to_ingreds_only[self.cocktail_index_to_name[i[0]]],
            'flavors': self.cocktail_names_to_flavors[self.cocktail_index_to_name[i[0]]],
            'popularity': self.cocktail_names_to_popularity[self.cocktail_index_to_name[i[0]]],
            'id': i[0]
        } for i in cos_rank]

        # boolean
        and_list = self.boolean_search_and(flavor_include_vec, matrix)
        not_list = self.boolean_search_not(flavor_exclude_vec, matrix)
        idx_list = list(set(and_list).intersection(set(not_list)))

        matrix = matrix[idx_list]
        rank_list = [
            i for i in rank_list
            if self.cocktail_name_to_index[i['name']] in idx_list]

        # print(drink_name_list)
        print(f"{len(rank_list)} drinks returned")
        print(cos_rank[0:5])
        
        # # display error message if invalid ingredient
        # if math.isnan(cos_rank[0][1]):
        #     return [{
        #         'id': -1,
        #         'name': "Please enter a valid ingredient or flavor.",
        #         'ingredients': "",
        #         'flavors': "",
        #         'popularity': ","
        #     }]
        return rank_list


# here for testing purposes (run $ python cocktailLab.py)
if __name__ == "__main__":
    cocktail = CocktailLab()
    print(cocktail.cocktail_names_to_popularity)
