import sys
import re
import string
import os
import numpy as np
import codecs
import pickle
import spacy

# Load spaCy's English model
nlp = spacy.load("en_core_web_sm")

# Get spaCy's English stop words
STOP_WORDS = nlp.Defaults.stop_words

def load_glove(filename):
    """
    Reads all lines from the indicated file and returns a dictionary
    mapping word:vector where vectors are of numpy `array` type.
    GloVe file lines are of the form:

    the 0.418 0.24968 -0.41242 0.1217 ...

    So split each line on spaces into a list; the first element is the word
    and the remaining elements represent factor components. The length of the vector
    should not matter; read vectors of any length.

    ignore stopwords
    """
    glove_dict = {}

    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.split()
            word = parts[0]
            if word not in STOP_WORDS:
                glove_dict[word] = np.array(parts[1:], dtype=float)
                
    return glove_dict


def filelist(root):
    """Returns a fully-qualified list of filenames under root directory"""
    allfiles = []
    for path, subdirs, files in os.walk(root):
        for name in files:
            allfiles.append(os.path.join(path, name))
    return allfiles


def get_text(filename):
    """
    Loads and returns the text of a text file, assuming latin-1 encoding as that
    is what the BBC corpus uses.
    """
    f = codecs.open(filename, encoding='latin-1', mode='r')
    s = f.read()
    f.close()
    return s


def words(text):
    """    
    Given a string, returns a list of words normalized using spaCy's tokenizer.
    - Lowercase all words
    - Filter out stop words and short words (< 3 characters)
    """
    doc = nlp(text.lower())
    return [token.text for token in doc if token.is_alpha and not token.is_stop and len(token.text) >= 3]


def split_title(text):
    """
    Returns title and the rest of the article.

    Splits the test by "\n" and assume that the first element is the title
    """
    parts = text.split("\n\n", 1)
    title = parts[0].strip()
    article = parts[1].strip() if len(parts) > 1 else ""

    return title, article


def load_articles(articles_dirname, gloves):
    """
    Loads all .txt files under articles_dirname and return a table (list of lists/tuples)
    where each record is a list of:

      [filename, title, article-text-minus-title, wordvec-centroid-for-article-text]

    Where filename represents the fully-qualified name of the text file including
    the path to the root of the corpus passed in on the command line.
    """
    articles_table = []
    
    for filepath in filelist(articles_dirname):
        if filepath.endswith(".txt"):
            filename_full = os.path.join(articles_dirname, filepath)
            with open(filepath, "r", encoding="latin-1") as f:
                text = f.read()
                title, article = split_title(text)
                wordvec_centroid = doc2vec(article, gloves)
                articles_table.append([filename_full, title, article, wordvec_centroid])
    return articles_table


def doc2vec(text, gloves):
    """
    Returns the word vector centroid for the text. Sums the word vectors
    for each word and then divides by the number of words. Ignores words
    not in gloves.
    """
    words_list = words(text)
    valid_vectors = [gloves[word] for word in words_list if word in gloves]

    if not valid_vectors:
        return np.zeros(len(next(iter(gloves.values()))))
    
    return np.mean(valid_vectors, axis=0)


def distances(article, articles):
    """
    Computes the euclidean distance from article to every other article.

    Inputs:
        article = [filename, title, text-minus-title, wordvec-centroid]
        articles is a list of [filename, title, text-minus-title, wordvec-centroid]

    Output:
        list of (distance, a) for a in articles
        where a is a list [filename, title, text-minus-title, wordvec-centroid]
    """
    distances_list = []
    article_vector = article[3]
    
    for other_article in articles:
            other_vector = other_article[3]
            distance = np.linalg.norm(article_vector - other_vector)  # Euclidean distance
            distances_list.append((distance, other_article))
    
    return distances_list


def recommended(article, articles, n):
    """ Return top n articles closest to article.

    Inputs:
        article: list [filename, title, text-minus-title, wordvec-centroid]
        articles: list of list [filename, title, text-minus-title, wordvec-centroid]

    Output:
         list of [topic, filename, title]
    """
    # Avoid self-recommendation
    current_title = article[1]
    seen_titles = {current_title}

    other_articles = [a for a in articles if a[0] != article[0]]

    dist_list = distances(article, other_articles)
    
    sorted_articles = sorted(dist_list, key=lambda x: x[0])
    
    recommendations = []
    for _, a in sorted_articles:
        topic = os.path.basename(os.path.dirname(a[0]))  # Extract topic
        filename = os.path.basename(a[0])  # Extract filename
        title = a[1]
        
        # Only add this article if we haven't seen its title before
        if title not in seen_titles:
            recommendations.append([topic, filename, title])
            seen_titles.add(title)
        
        # Stop if we have n recommendations
        if len(recommendations) == n:
            break
    
    return recommendations


def main():
    if len(sys.argv) != 3:
        print("Usage: python script.py <glove_filename> <articles_dirname>")
        sys.exit(1)

    glove_filename = sys.argv[1]
    articles_dirname = sys.argv[2]

    try:
        gloves = load_glove(glove_filename)
        articles = load_articles(articles_dirname, gloves)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)
    
    # Save articles data in a pickle file
    # [[topic, filename, title, text], ...]
    try:
        with open('./processed/pickles/articles.pkl', 'wb') as f:
            articles_data = [[os.path.basename(os.path.dirname(a[0])),  # Extract topic
                              os.path.basename(a[0]),  # Extract filename
                              a[1],                   # Title
                              a[2]]                   # Article text
                              for a in articles]
            pickle.dump(articles_data, f)
    except Exception as e:
        print(f"Error saving articles data: {e}")

    # save in recommended.pkl a dictionary with top 5 recommendations for each article. 
    # given an article use (topic, filename) as the key
    # the recommendations are a list of [topic, filename, title] for the top 5 closest articles
    recommendations = {}
    for article in articles:
        recs = recommended(article, articles, 5)
        topic = os.path.basename(os.path.dirname(article[0]))  # Extract topic
        filename = os.path.basename(article[0])  # Extract filename
        recommendations[(topic, filename)] = recs
        
    try:
        with open('./processed/pickles/recommended.pkl', 'wb') as f:
            pickle.dump(recommendations, f)
    except Exception as e:
        print(f"Error saving recommendations: {e}")


if __name__ == '__main__':    
    main()