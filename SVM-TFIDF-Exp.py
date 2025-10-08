import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

train_data = pd.read_csv(r"C:\Users\Lenovo\Downloads\LREC-Data\LREC-Data\TRAIN_Medical_Condition_Gemini_2.5_pro_segmented.csv",encoding='utf-8')
test_data = pd.read_csv(r"C:\Users\Lenovo\Downloads\LREC-Data\LREC-Data\TEST_Medical_Condition_Gemini_2.5_pro_segmented.csv",encoding='utf-8')




def tfidf(data):
    tfidf_vectorizer = TfidfVectorizer()

    train = tfidf_vectorizer.fit_transform(data)

    return train, tfidf_vectorizer


X_train = train_data["word"].tolist()
Y_train = train_data["labels"].tolist()
X_test = test_data["translated_word"].tolist()
Y_test = test_data["labels"].tolist()

X_train_tfidf, tfidf_vectorizer = tfidf(X_train)
print(tfidf_vectorizer.get_feature_names_out())
X_test_tfidf = tfidf_vectorizer.transform(X_test)

clf = svm.LinearSVC()
clf.fit(X_train_tfidf, Y_train)
y_predicted_tfidf = clf.predict(X_test_tfidf)
print('accuracy %s' % accuracy_score(y_predicted_tfidf, test_data.labels))
print('F1 %s' % f1_score(y_predicted_tfidf, test_data.labels, average='weighted'))
print("Classification Report By TF-IDF")
report = classification_report(Y_test, y_predicted_tfidf)
print(report)
print("Confusion Matrix TFIDF")
print(confusion_matrix(Y_test, y_predicted_tfidf))
