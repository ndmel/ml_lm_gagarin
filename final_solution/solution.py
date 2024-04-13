import typing as tp


EntityScoreType = tp.Tuple[int, float]  # (entity_id, entity_score)
MessageResultType = tp.List[
    EntityScoreType
]  # list of entity scores,
#    for example, [
#       (entity_id, entity_score)
#       for entity_id, entity_score in entities_found
#    ]


def score_texts(
    messages: tp.Iterable[str], *args, **kwargs
) -> tp.Iterable[MessageResultType]:
    """
    Main function (see tests for more clarifications)
    Args:
        messages (tp.Iterable[str]): any iterable of strings
        (utf-8 encoded text messages)

    Returns:
        tp.Iterable[tp.Tuple[int, float]]: for any messages
        returns MessageResultType object
    -------
    Clarifications:
    >>> assert all([len(m) < 10 ** 11 for m in messages])
    # all messages are shorter than 2048 characters
    """

    vectorizer_company, vectorizer_sentiment, clf_company, clf_sentiment = args

    companies = [int(x) for x in clf_company.predict(vectorizer_company.transform(messages))]
    sentiments = [float(x) for x in clf_sentiment.predict(vectorizer_sentiment.transform(messages))]
    result = [[pair] for pair in zip(companies, sentiments)]

    return result
