# Article Recommendation System

![Python](https://img.shields.io/badge/Python-blue)
![Flask](https://img.shields.io/badge/Flask-orange)
![NumPy](https://img.shields.io/badge/NumPy-red)
![pandas](https://img.shields.io/badge/pandas-green)
![lxml](https://img.shields.io/badge/lxml-lightgrey)
![lxml](https://img.shields.io/badge/spacy-lightblue)

This project implements an article recommendation system using word vectors and a web application to display the results.

## Goal

Develop and improve upon an article recommendation engine utilizing word vectors through the [word2vec](http://arxiv.org/pdf/1301.3781.pdf) technique. It leverages a "database" of word vectors (embeddings) from [Stanford's GloVe project](https://nlp.stanford.edu/projects/glove/) trained on a Wikipedia dump. The project involves reading a database of word vectors and a corpus of text articles from the BBC, then organizing them into a structured format for efficient processing.

The application features a web server displaying a list of articles from the [BBC](http://mlg.ucd.ie/datasets/bbc.html) dataset, accessible at `http://127.0.0.1:5000`.

Run `server.py` to run the `Flask` app locally.

```bash
python server.py
```

Access the homepage at the URL provided in `IP.txt`:

```
http://127.0.0.1:5000/
```

<img src="./figures/articles.png" alt="Articles List" width="400" />

Clicking on an article will take you to a dedicated page showing the article text and a list of five recommended articles.

<img src="./figures/article1.png" alt="Articles List" width="700" />

You can also search for a specific article by specifying in the URL:

```
http://127.0.0.1:5000/article/tech/076.txt
```

<img src="./figures/article2.png" alt="Articles List" width="700" />

## Future Improvements for the Flask App

### 1. Enhanced User Interface
- **Responsive Design**: Use Bootstrap to make the app visually appealing and responsive, to improve user experience.

### 2. Improved Article Recommendations
- **Personalized Recommendations**: Implement a simple user feedback mechanism, thumbs up/down and favorite buttons for articles.

### 3. Advanced Search Functionality
- **Search Bar**: Add a search bar to allow users to search for articles by keywords.

## Setup
Download the following data in `~/data` directory:

```bash
wget https://s3-us-west-1.amazonaws.com/msan692/glove.6B.300d.txt.zip
wget https://s3-us-west-1.amazonaws.com/msan692/bbc.zip
```

## Part 1: Database of Recommendations

### Components

Run `doc2vec.py` scripts to get the pickle files.

```bash
python doc2vec.py ~/data/glove.6B.300d.txt ~/data/bbc
```

Should generate `articles.pkl` and `recommended.pkl` which are used for processing:
* `articles.pkl`: Contains a list of articles, where each article is represented as a list with key information:

  ```python
  [[topic, filename, title, text], ...]
  ```
* `recommended.pkl`: Stores a dictionary where each key is a tuple of (topic, filename), and each value is a list of recommended articles:
  ```python
  {
    (topic, filename): [
        [topic, filename, title],
        [topic, filename, title],
        [topic, filename, title],
        [topic, filename, title],
        [topic, filename, title],
    ],
  ...
  )
  ```

## Part 2: Web Application Development

### Components

1. `server.py`: Implements Flask routes to handle HTTP requests.
2. `templates/articles.html`: Uses template language to generate HTML for the main articles list page.
3. `templates/article.html`: Uses template language to generate HTML for individual article pages.

## Discussion

### Article Word-Vector Centroids

Each word is represented by a 300-dimensional vector capturing its meaning, derived from a neural network that organizes words based on their contexts. Articles' centroids can be computed using these vectors, allowing for effective similarity measurements between documents.

### Efficiently Loading the GloVe File

To optimize memory usage, `doc2vec.py` reads the GloVe file line by line, building a dictionary incrementally.

### Web Server Implementation

The Flask server should handle two primary URLs: a list of articles at `/` and individual articles at `/article/topic/filename`. The BBC corpus is organized into topic directories containing text files.

The `server.py` file contains Flask route definitions:

```python
@app.route("/")
```

```python
@app.route("/article/<topic>/<filename>")
```

It also utilizes the Jinja2 template engine for rendering HTML templates.

## Dependencies

This project requires the following packages:

```bash
pip install Flask
pip install numpy
pip install pandas
pip install lxml
pip install spacy
```
