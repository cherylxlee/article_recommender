# Article Recommendation System

![Python](https://img.shields.io/badge/Python-blue)
![Flask](https://img.shields.io/badge/Flask-orange)
![NumPy](https://img.shields.io/badge/NumPy-red)
![pandas](https://img.shields.io/badge/pandas-green)
![lxml](https://img.shields.io/badge/lxml-lightgrey)

This project implements an article recommendation system using word vectors and a web application to display the results.

## Goal

The goal of this project is to develop an article recommendation engine utilizing word vectors through the [word2vec](http://arxiv.org/pdf/1301.3781.pdf) technique. It leverages a "database" of word vectors from [Stanford's GloVe project](https://nlp.stanford.edu/projects/glove/) trained on a Wikipedia dump. The project involves reading a database of word vectors and a corpus of text articles, then organizing them into a structured format for efficient processing.

The application will feature a web server displaying a list of articles from the [BBC](http://mlg.ucd.ie/datasets/bbc.html) dataset, accessible at `http://127.0.0.1:5000`.

![Articles List](./figures/articles.png)

Clicking on an article will take you to a dedicated page showing the article text and a list of five recommended articles.

![Article Page](./figures/article1.png)
![Article Recommendations](./figures/article2.png)

## Part 1: Database of Recommendations

### Components

1. `doc2vec.py`: Complete the implementation of the functions defined in this file.

To validate this section, run the following test:

```bash
pytest -v test_doc2vec.py
```

Executing

```bash
python doc2vec.py ~/data/glove.6B.300d.txt ~/data/bbc
```

Should generate `articles.pkl` and `recommended.pkl`, which will be utilized in Part 2.

## Part 2: Web Application Development

### Components

1. `server.py`: Implement Flask routes to handle HTTP requests.
2. `templates/articles.html`: Use template language to generate HTML for the main articles list page.
3. `templates/article.html`: Use template language to generate HTML for individual article pages.

To validate this part, run:

```bash
pytest test_server.py
```

Ensure your application is running locally to execute the tests.

## Discussion

### Article Word-Vector Centroids

Each word is represented by a 300-dimensional vector capturing its meaning, derived from a neural network that organizes words based on their contexts. Articles' centroids can be computed using these vectors, allowing for effective similarity measurements between documents.

### Efficiently Loading the GloVe File

To optimize memory usage, read the GloVe file line by line, building a dictionary incrementally:

```python
d = {}
for line in f:
    # Process line to build the dictionary
```

### Web Server Implementation

The Flask server should handle two primary URLs: a list of articles at `/` and individual articles at `/article/topic/filename`. The BBC corpus is organized into topic directories containing text files.

When testing, access the articles list via:

```
http://127.0.0.1:5000/
```

And a specific article at:

```
http://127.0.0.1:5000/article/business/030.txt
```

The `server.py` file will contain Flask route definitions that you need to complete:

```python
@app.route("/")
def articles():
    """Show a list of article titles"""
```

```python
@app.route("/article/<topic>/<filename>")
def article(topic, filename):
    """Show an article with the given filename."""
```

Utilize the Jinja2 template engine for rendering HTML templates.

## Dependencies

This project requires the following packages:

```bash
pip install Flask
pip install numpy
pip install pandas
pip install lxml
```
