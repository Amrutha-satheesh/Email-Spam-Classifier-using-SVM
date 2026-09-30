from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

# 1. Your Email Data (Text)
emails = [
    "Free entry to win a prize cash reward",      # Spam
    "Are you coming for dinner tonight?",         # Not Spam (Ham)
    "Urgent! Claim your free gift card now",       # Spam
    "Let's schedule the meeting for Monday"       # Not Spam (Ham)
]

# 2. Labels (0 = Not Spam, 1 = Spam)
y = np.array([1, 0, 1, 0])

# 3. Text to Numbers (Vectorization)
# This replaces your manual numpy array creation.
# It creates a grid where each column is a word and each row is an email.
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails).toarray()

# 4. Define and Train the SVM Model
model = SVC(kernel='linear') 
model.fit(X, y)

# 5. Predict on a New Email
new_email = ["Win a free prize tonight"]
new_email_converted = vectorizer.transform(new_email).toarray()

prediction = model.predict(new_email_converted)

# 6. Output Result
result = "Spam" if prediction[0] == 1 else "Not Spam"
print(f"Prediction for '{new_email[0]}': {result}")