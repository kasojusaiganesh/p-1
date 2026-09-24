
from flask import Flask, render_template, request, jsonify

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

# Training emails
emails = [
    "Congratulations! You won a free lottery prize",
    "Claim your free gift card now",
    "You have won $1000, click here",
    "Limited offer! Get free money today",
    "Win a brand new phone by entering now",
    "Urgent! You have received a cash reward",

    "Hi, how are you doing today?",
    "Please find the assignment attached",
    "Meeting is scheduled for tomorrow",
    "Can you send me the project report?",
    "Your college class starts at 9 AM",
    "Let's have lunch together today"
]

# Spam = 1, Genuine = 0
labels = [
    1, 1, 1, 1, 1, 1,
    0, 0, 0, 0, 0, 0
]

# Train the ML model
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(emails)

model = MultinomialNB()
model.fit(X, labels)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    new_email = data.get("email", "").strip()

    if not new_email:
        return jsonify({
            "error": "Please enter an email message."
        }), 400

    email_features = vectorizer.transform([new_email])

    prediction = model.predict(email_features)[0]

    probabilities = model.predict_proba(email_features)[0]
    confidence = probabilities[prediction] * 100

    if prediction == 1:
        result = "SPAM EMAIL"
    else:
        result = "GENUINE EMAIL"

    return jsonify({
        "result": result,
        "confidence": round(confidence, 2)
    })


if __name__ == "__main__":
    app.run(debug=True)