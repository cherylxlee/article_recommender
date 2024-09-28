# Launch with
#
# python app.py

from flask import Flask, render_template
import sys
import pickle

app = Flask(__name__)

@app.route("/")
def articles():
    """Show a list of article titles"""
    ## implement


@app.route("/article/<topic>/<filename>")
def article(topic, filename):
    """
    Show an article with relative path filename. Assumes the BBC structure of
    topic/filename.txt so our URLs follow that.
    """
    ## implement


f = open('articles.pkl', 'rb')
articles = pickle.load(f)
f.close()

f = open('recommended.pkl', 'rb')
recommended = pickle.load(f)
f.close()


# for local debug
if __name__ == '__main__':
    app.run(debug=True)

