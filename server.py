# Launch with
#
# python app.py

from flask import Flask, render_template
import sys
import pickle
import pandas

app = Flask(__name__)

@app.route("/")
def articles():
    """Show a list of article titles"""
    articles_sorted = sorted(articles, key=lambda x: (x[0], x[1]))

    grouped_articles = {}
    for article in articles_sorted:
        topic = article[0]
        title = article[2]

        if topic not in grouped_articles:
            grouped_articles[topic] = {}

        if title not in grouped_articles[topic]:
            grouped_articles[topic][title] = {
                "filename": article[1],
                "title": title
            }

    for topic in grouped_articles:
        grouped_articles[topic] = list(grouped_articles[topic].values())

    return render_template("articles.html", grouped_articles=grouped_articles)


@app.route("/article/<topic>/<filename>")
def article(topic, filename):
    """
    Show an article with relative path filename. Assumes the BBC structure of
    topic/filename.txt so our URLs follow that.
    """
    article_key = (topic, filename)

    article_data = next((article for article in articles if article[0] == topic and article[1] == filename), None)
    
    if article_data is None:
        return render_template('404.html'), 404

    title = article_data[2]
    text = article_data[3].split("\n\n")
    recs = recommended.get((topic, filename), [])

    return render_template('article.html', title=title, text=text, recommendations=recs)


f = open('./processed/pickles/articles.pkl', 'rb')
articles = pickle.load(f)
f.close()

f = open('./processed/pickles/recommended.pkl', 'rb')
recommended = pickle.load(f)
f.close()


# for local debug
if __name__ == '__main__':
    app.run(debug=True)

