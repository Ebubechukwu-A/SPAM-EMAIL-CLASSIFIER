
import csv
import math
import re
from collections import defaultdict






# Process the text using .lower, re.findall()

def tokenize(text: str) -> list[str]:
    """Lowercase and split text into word tokens, stripping punctuation."""
    text = text.lower()
    tokens = re.findall(r"[a-z]+", text)
    return tokens


#  Naive Bayes Classifier 

class NaiveBayesClassifier:
  

    def __init__(self):
        self.class_log_prior: dict[str, float] = {}
        self.word_log_likelihood: dict[str, dict[str, float]] = {}
        self.classes: list[str] = []
        self.vocab: set[str] = set()
        # Raw counts (exposed for transparency / report)
        self._class_word_counts: dict[str, dict[str, int]] = {}
        self._class_total_words: dict[str, int] = {}
        self._class_doc_counts: dict[str, int] = {}

    #  Training 
    def train(self, dataset: list[tuple[str, str]]) -> None:   #Setting up the formattinng for the csv's

       

        word_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))   # how many emails per class
        doc_counts: dict[str, int] = defaultdict(int)     #how many times each word appears per class
        total_docs = len(dataset)

        for text, label in dataset:   #Looping through emails, I've now seen 450 spam emails and 550 ham emails
            doc_counts[label] += 1
            for token in tokenize(text):      #Break email into words, The word 'free' has appeared 90 times in spam, 2 times in ham

                word_counts[label][token] += 1
                self.vocab.add(token)

        self.classes = list(doc_counts.keys())
        vocab_size = len(self.vocab)



#------------------------------------------------------




        # CONVERTING RAW COUNTS INTO PROBABBILTIY (caclucations)
        for cls in self.classes:
            # Log prior:  log P(class)
            self.class_log_prior[cls] = math.log(doc_counts[cls] / total_docs)

            total_words = sum(word_counts[cls].values())
            self._class_word_counts[cls] = dict(word_counts[cls])   #Comparing word vs amount of total words to get the score
            self._class_total_words[cls] = total_words
            self._class_doc_counts[cls] = doc_counts[cls]

            # Log likelihood with Laplace smoothing:

            #   log P(word | class) = log( (count(word,class) + 1) /
            #                              (total_words_in_class + vocab_size) )
            self.word_log_likelihood[cls] = {}
            denominator = total_words + vocab_size
            for word in self.vocab:
                count = word_counts[cls].get(word, 0)
                self.word_log_likelihood[cls][word] = math.log(
                    (count + 1) / denominator
                )

            # Store the smoothed unknown-word likelihood for unseen tokens
            self.word_log_likelihood[cls]["<UNK>"] = math.log(1 / denominator)



    # Prediction func 
    # Actually uses the porbablilties from "for cls in self.classses"

    def predict(self, text: str) -> dict:

        tokens = tokenize(text)


        #Actual naive bayes forumals

        log_posteriors: dict[str, float] = {}
        for cls in self.classes:
            score = self.class_log_prior[cls]    #Prior probabiloity formula (comparing word vs amount of normal emails) aka log P(class)
            for token in tokens:    
                if token in self.word_log_likelihood[cls]:
                    score += self.word_log_likelihood[cls][token]  # + log P(word | class) , the "score" of the word we calcualted from for "cls in self.classes"
                else:
                    # Unseen word, use smoothed unknown likelihood (add 1 to words that AI hasnt seen before)
                    score += self.word_log_likelihood[cls]["<UNK>"]
            log_posteriors[cls] = score  #final score


        # Convert log-posteriors to percentages for humans to read
        max_score = max(log_posteriors.values())
        exp_scores = {cls: math.exp(s - max_score) for cls, s in log_posteriors.items()}
        total = sum(exp_scores.values())
        probabilities = {cls: round(v / total, 4) for cls, v in exp_scores.items()}

        prediction = max(log_posteriors, key=log_posteriors.get)

        return {
            "prediction": prediction,
            "scores": {cls: round(s, 4) for cls, s in log_posteriors.items()},
            "probabilities": probabilities,
            "tokens_used": tokens,
        }




#------------------------------------------------------


    # Confusion matrix, a scoreboard of right and wrong answers


    def evaluate(self, dataset: list[tuple[str, str]]) -> dict:
   
        labels = self.classes
        # confusion_matrix[actual][predicted]
        cm: dict[str, dict[str, int]] = {
            cls: {c: 0 for c in labels} for cls in labels
        }

        for text, actual in dataset:
            predicted = self.predict(text)["prediction"]
            cm[actual][predicted] += 1

        metrics_per_class = {}
        for cls in labels:
            tp = cm[cls][cls]
            fp = sum(cm[other][cls] for other in labels if other != cls)
            fn = sum(cm[cls][other] for other in labels if other != cls)

            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall    = tp / (tp + fn) if (tp + fn) else 0.0
            f1        = (2 * precision * recall / (precision + recall)
                         if (precision + recall) else 0.0)

            metrics_per_class[cls] = {
                "precision": round(precision, 4),
                "recall":    round(recall,    4),
                "f1":        round(f1,        4),
            }

        total = len(dataset)
        correct = sum(cm[cls][cls] for cls in labels)
        accuracy = round(correct / total, 4) if total else 0.0

        return {
            "accuracy": accuracy,
            "per_class": metrics_per_class,
            "confusion_matrix": cm,
        }


    def top_words(self, cls: str, n: int = 10) -> list[tuple[str, int]]:
        counts = self._class_word_counts.get(cls, {})
        return sorted(counts.items(), key=lambda x: x[1], reverse=True)[:n]

    def vocab_size(self) -> int:
        return len(self.vocab)


def load_dataset_from_csv(filename: str) -> list[tuple[str, str]]:
    dataset = []
    with open(filename, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            text = row["text"].strip()
            label = row["label"].strip().lower()
            dataset.append((text, label))
    return dataset




#------------------------------------------------------
_classifier = NaiveBayesClassifier()
TRAINING_DATA = load_dataset_from_csv("train_emails.csv")  # loading data
TEST_DATA = load_dataset_from_csv("test_emails.csv")  # loading data
_classifier.train(TRAINING_DATA)

def classify_email(text: str) -> dict:
    """Public function: classify a single email and return result dict."""
    return _classifier.predict(text)


def get_evaluation_report() -> dict:
    """Evaluate classifier on training data (use a held-out set in production)."""
    return _classifier.evaluate(TEST_DATA)  # evaluating on unseen test data csv




#------------------------------------------------------
def get_model_info() -> dict:
    #Return summary info about the trained model
    return {
        "vocabulary_size": _classifier.vocab_size(),
        "classes": _classifier.classes,
        "training_samples": len(TRAINING_DATA),
        "test_samples": len(TEST_DATA),  
        "class_distribution": _classifier._class_doc_counts,
        "top_spam_words": _classifier.top_words("spam"),
        "top_ham_words":  _classifier.top_words("ham"),
    }

#------------------------------------------------------

#Self-test based on test data csv

if __name__ == "__main__":
    test_emails = [
        "Click here to claim your free prize now",
        "Can we reschedule our meeting to Thursday",
        "Exclusive offer just for you win big money",
        "The assignment is due next Monday at midnight",
    ]

    print("=== Naive Bayes Spam Classifier ===\n")

    for email in test_emails:
        result = classify_email(email)
        prob = result["probabilities"]
        print(f"Email : {email}")
        print(f"Result: {result['prediction'].upper()}  "
              f"(spam={prob.get('spam', 0):.2%}, ham={prob.get('ham', 0):.2%})")
        print()

    print("=== Evaluation Report ===")
    report = get_evaluation_report()
    print(f"Accuracy : {report['accuracy']:.2%}")
    for cls, m in report["per_class"].items():
        print(f"{cls.capitalize():5s} — Precision: {m['precision']:.2%}  "
              f"Recall: {m['recall']:.2%}  F1: {m['f1']:.2%}")

    print("\n=== Confusion Matrix ===")
    cm = report["confusion_matrix"]
    classes = _classifier.classes
    header = f"{'':10s}" + "".join(f"{c:>10s}" for c in classes)
    print(header)
    for actual in classes:
        row = f"{actual:10s}" + "".join(f"{cm[actual][pred]:>10d}" for pred in classes)
        print(row)

    print("\n=== Model Info ===")
    info = get_model_info()
    print(f"Vocabulary size : {info['vocabulary_size']}")
    print(f"Training samples: {info['training_samples']}")
    print(f"Test samples    : {info['test_samples']}")  
    print(f"Top spam words  : {[w for w, _ in info['top_spam_words'][:5]]}")
    print(f"Top ham words   : {[w for w, _ in info['top_ham_words'][:5]]}")