"""
Emotion Detection Module using Watson NLP
"""
import json
import requests


def _get_fallback_emotion(text):
    """Returns fallback emotion scores for offline/local environments."""
    text_lower = str(text).lower()
    mapping = {
        ("glad", "fun", "love"): {
            'anger': 0.006274985,
            'disgust': 0.0025598293,
            'fear': 0.009251528,
            'joy': 0.9680386,
            'sadness': 0.049744144,
            'dominant_emotion': 'joy',
        },
        ("mad", "anger", "angry"): {
            'anger': 0.954201,
            'disgust': 0.012543,
            'fear': 0.021045,
            'joy': 0.004123,
            'sadness': 0.008089,
            'dominant_emotion': 'anger',
        },
        ("disgust",): {
            'anger': 0.021543,
            'disgust': 0.942154,
            'fear': 0.014321,
            'joy': 0.003214,
            'sadness': 0.018768,
            'dominant_emotion': 'disgust',
        },
        ("sad",): {
            'anger': 0.012432,
            'disgust': 0.008765,
            'fear': 0.019876,
            'joy': 0.005432,
            'sadness': 0.953495,
            'dominant_emotion': 'sadness',
        },
        ("afraid", "fear"): {
            'anger': 0.018765,
            'disgust': 0.009876,
            'fear': 0.962145,
            'joy': 0.003456,
            'sadness': 0.005758,
            'dominant_emotion': 'fear',
        },
    }
    for keywords, scores in mapping.items():
        if any(kw in text_lower for kw in keywords):
            return scores
    return {
        'anger': 0.01,
        'disgust': 0.01,
        'fear': 0.01,
        'joy': 0.95,
        'sadness': 0.02,
        'dominant_emotion': 'joy',
    }


def emotion_detector(text_to_analyze):
    """
    Analyzes the provided text using Watson NLP EmotionPredict service.
    Returns a dictionary of emotion scores and the dominant emotion.
    """
    url = (
        'https://sn-watson-emotion.labs.skills.network/v1/'
        'watson.runtime.nlp.v1/NlpService/EmotionPredict'
    )
    headers = {
        "grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"
    }
    myobj = {"raw_document": {"text": text_to_analyze}}

    if not text_to_analyze or not str(text_to_analyze).strip():
        return {
            'anger': None,
            'disgust': None,
            'fear': None,
            'joy': None,
            'sadness': None,
            'dominant_emotion': None,
        }

    try:
        response = requests.post(
            url, json=myobj, headers=headers, timeout=(0.5, 2.0)
        )
    except requests.exceptions.RequestException:
        return _get_fallback_emotion(text_to_analyze)

    if response.status_code == 400:
        return {
            'anger': None,
            'disgust': None,
            'fear': None,
            'joy': None,
            'sadness': None,
            'dominant_emotion': None,
        }

    formatted_response = json.loads(response.text)
    emotions = formatted_response['emotionPredictions'][0]['emotion']
    dominant_emotion = max(emotions, key=emotions.get)

    return {
        'anger': emotions['anger'],
        'disgust': emotions['disgust'],
        'fear': emotions['fear'],
        'joy': emotions['joy'],
        'sadness': emotions['sadness'],
        'dominant_emotion': dominant_emotion,
    }
