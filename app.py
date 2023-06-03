import nltk
import os
from flask import Flask, render_template, request

app = Flask(__name__)

def summarize_text(text, num_sentences=3):
    sentences = nltk.sent_tokenize(text)
    words = nltk.word_tokenize(text)
    freq_dist = nltk.FreqDist(words)
    most_frequent_words = [word for word, _ in freq_dist.most_common(10)]

    sentence_scores = {}
    for sentence in sentences:
        for word in nltk.word_tokenize(sentence.lower()):
            if word in most_frequent_words:
                if sentence not in sentence_scores:
                    sentence_scores[sentence] = 1
                else:
                    sentence_scores[sentence] += 1

    sorted_sentences = sorted(
        sentence_scores.items(), key=lambda x: x[1], reverse=True
    )
    top_sentences = [sentence for sentence, _ in sorted_sentences[:num_sentences]]

    summary = " ".join(top_sentences)
    return summary


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/summarize", methods=["POST"])
def summarize():
    text = request.form["text"]
    summary = summarize_text(text)
    return render_template("summary.html", summary=summary)


if __name__ == "__main__":
    nltk_path = os.path.join(os.getcwd(), "nltk_data")
    nltk.data.path.append(nltk_path)
    nltk.download("punkt", download_dir=nltk_path)
    app.run(debug=True)
