import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

#Sample training data
texts = [
    "I was overcharged on my invoice",
    "Refund not processed",
    "System crashes when I login",
    "API integration not working",
    "Need contract termination details",
    "Legal compliance issue with agreement"
]

labels = [
    "Billing",
    "Billing",
    "Technical",
    "Technical",
    "Legal",
    "Legal"
]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

model = LogisticRegression()
model.fit(X, labels)

with open("model.pkl", "wb") as f:
    pickle.dump((vectorizer, model), f)

print("Model trained and saved.")
