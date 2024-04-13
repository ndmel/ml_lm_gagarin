import json
import pathlib
import typing as tp
import joblib

from final_solution.solution import score_texts


PATH_TO_TEST_DATA = pathlib.Path('data') / 'test_texts.json'
PATH_TO_OUTPUT_DATA = pathlib.Path('results') / 'output_scores.json'
PATH_TO_COMPANY_VECTORIZER = pathlib.Path('model') / 'vectorizer_company.pkl'
PATH_TO_SENTIMENT_VECTORIZER = pathlib.Path('model') / 'vectorizer_sentiment.pkl'
PATH_TO_COMPANY_MODEL = pathlib.Path('model') / 'clf_company.pkl'
PATH_TO_SENTIMENT_MODEL = pathlib.Path('model') / 'clf_sentiment.pkl'


def load_artifacts(
    vector_company_path: pathlib.Path = PATH_TO_COMPANY_VECTORIZER,
    vector_sentiment_path: pathlib.Path = PATH_TO_SENTIMENT_VECTORIZER,
    company_model_path: pathlib.Path = PATH_TO_COMPANY_MODEL,
    sentiment_model_path: pathlib.Path = PATH_TO_SENTIMENT_MODEL
):

    vectorizer_company = joblib.load(vector_company_path)
    vectorizer_sentiment = joblib.load(vector_sentiment_path)
    clf_company = joblib.load(company_model_path)
    clf_sentiment = joblib.load(sentiment_model_path)

    return vectorizer_company, vectorizer_sentiment, clf_company, clf_sentiment


def load_data(path: pathlib.Path = PATH_TO_TEST_DATA) -> tp.List[str]:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data


def save_data(data, path: pathlib.Path = PATH_TO_OUTPUT_DATA):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=1, ensure_ascii=False)


def main():
    texts = load_data()
    vectorizer_company, vectorizer_sentiment, clf_company, clf_sentiment = load_artifacts()
    scores = score_texts(
        texts, vectorizer_company, vectorizer_sentiment, clf_company, clf_sentiment
    )
    save_data(scores)


if __name__ == '__main__':
    main()
