
import json
import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from collections import Counter
from langdetect import detect
from nltk.corpus import stopwords

def clean_tweet(text):
    """
    Nettoie le tweet en retirant les hashtags, les mentions, les liens, les chiffres, la ponctuation et les stopwords.
    """
    stop_words = set(stopwords.words('english') + stopwords.words('spanish'))
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = ' '.join([word for word in text.split() if word.lower() not in stop_words])
    return text.strip()

def load_data(training_file):
    """
    Charge les données de "training.json", nettoie les tweets et renvoie les vecteurs de mots et les labels correspondants.
    """
    with open(training_file, 'r', encoding='utf-8') as f:
        training_data = json.load(f)

    tweets = []
    labels = []
    for tweet_id, tweet_data in training_data.items():
        tweet = tweet_data['tweet']
        try:
            if detect(tweet) == 'en' or detect(tweet) == 'es':
                clean = clean_tweet(tweet)
                tweets.append(clean)
                labels.append(1 if tweet_data['labels_task1'][0] == 'YES' else 0)
        except:
            pass

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(tweets)
    y = np.array(labels)

    return X, y, vectorizer

def train_model(training_file):
    """
    Entraîne un modèle de classification SVM sur les données de "training.json".
    """
    X, y, vectorizer = load_data(training_file)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = LinearSVC()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    return clf, vectorizer

clf, vectorizer = train_model("training.json")

def get_sexist_words(training_file, result_file):
    """
    Lit les fichiers "training.json" et "res.json", nettoie les tweets dans "training.json",
    et renvoie la fréquence des mots dans les tweets marqués comme sexistes selon "res.json"
    avec un soft label "YES" supérieur à 50%.
    """
    with open(training_file, 'r',encoding='utf-8') as f:
        training_data = json.load(f)

    with open(result_file, 'r',encoding='utf-8') as f:
        result_data = json.load(f)

    sexist_tweets = []
    for tweet_id, result in result_data.items():
        if result['soft_label']['YES'] > 0.5:
            sexist_tweets.append(tweet_id)

    counter = Counter()
    for tweet_id, tweet_data in training_data.items():
        if tweet_id in sexist_tweets:
            tweet = tweet_data['tweet']
            try:
                if detect(tweet) == 'en' or detect(tweet) == 'es':
                    clean = clean_tweet(tweet)
                    counter.update(clean.split())
            except:
                pass
    return counter


def score_word(freq, max_freq):
    """
    Attribue un score entre 0 et 1 en fonction de la fréquence du mot.
    """
    score = freq / max_freq
    return score

def score_words(counter):
    """
    Attribue un score à chaque mot en fonction de sa fréquence dans les tweets sexistes.
    """
    max_freq = max(counter.values())
    scores = {}
    for word, freq in counter.items():
        scores[word] = score_word(freq, max_freq)
    return scores

counter = get_sexist_words("./training.json", "./training_task1_gold_soft.json")
scores = score_words(counter)

def is_sexist_tweet(tweet, clf, vectorizer):
    """
    Détermine si le tweet est sexiste ou non en utilisant le modèle entraîné.
    """
    clean = clean_tweet(tweet)
    vectorized = vectorizer.transform([clean])
    prediction = clf.predict(vectorized)[0]
    return prediction

tweets = [
    "Women are so stupid they can't even drive LOL",
    "How can this douchebag be president??",
    "Why is this dickhead present...",
    "Fuck off, I do whatever I want",
    "I love my wife, she's the best!!!",
    "Hello, can you help me with my homeworks?",
    "How can I open this jar?",
    "I will be here in 10 minutes kid."
]
test = [int(is_sexist_tweet(tweet, clf, vectorizer)) for tweet in tweets]
print(test)
with open ("./res.json", "w") as f:
    res = {}
    for i, result in enumerate(test):
        res[i] = {
            "hard_label": "YES" if result == 1 else "NO",
            "soft_label": {
                "YES": result,
                "NO": (1 - result)
            }
        }
    f.write(json.dumps(res))
