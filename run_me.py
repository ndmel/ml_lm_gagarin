import json
import pathlib
import typing as tp
import joblib

from final_solution.solution import score_texts


PATH_TO_TEST_DATA = pathlib.Path("data") / "test_texts.json"
PATH_TO_OUTPUT_DATA = pathlib.Path("results") / "output_scores.json"
PATH_TO_VECTORIZER = pathlib.Path("model") / "vectorizer.pkl"
PATH_TO_COMPANY_MODEL = pathlib.Path("model") / "clf_company.pkl"
PATH_TO_SENTIMENT_MODEL = pathlib.Path("model") / "clf_sentiment.pkl"


def load_artifacts(vector_path: pathlib.PosixPath = PATH_TO_VECTORIZER, 
                   company_model_path: pathlib.PosixPath = PATH_TO_COMPANY_MODEL, 
                   sentiment_model_path: pathlib.PosixPath = PATH_TO_SENTIMENT_MODEL ):
    
    
    vectorizer = joblib.load(vector_path)
    clf_company = joblib.load(company_model_path)
    clf_sentiment = joblib.load(sentiment_model_path)
    
    return vectorizer, clf_company, clf_sentiment


def load_data(path: pathlib.PosixPath = PATH_TO_TEST_DATA) -> tp.List[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data


def save_data(data, path: pathlib.PosixPath = PATH_TO_OUTPUT_DATA):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1, ensure_ascii=False)


def main():
    texts = load_data()
    vectorizer, clf_company, clf_sentiment = load_artifacts()
    scores = score_texts(texts, vectorizer, clf_company, clf_sentiment)
    save_data(scores)

if __name__ == '__main__':
    main()
