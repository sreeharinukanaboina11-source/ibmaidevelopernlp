"""
Unit tests for Emotion Detection application.
"""
import unittest
from EmotionDetection.emotion_detection import emotion_detector


class TestEmotionDetection(unittest.TestCase):
    """
    Test suite for emotion detection function.
    """

    def test_emotion_detector(self):
        """
        Tests dominant emotions for the 5 required statements:
        1. 'I am glad this happened' -> joy
        2. 'I am really mad about this' -> anger
        3. 'I feel disgusted just hearing about this' -> disgust
        4. 'I am so sad about this' -> sadness
        5. 'I am really afraid that this will happen' -> fear
        """
        # Test 1: joy
        result_1 = emotion_detector('I am glad this happened')
        self.assertEqual(result_1['dominant_emotion'], 'joy')

        # Test 2: anger
        result_2 = emotion_detector('I am really mad about this')
        self.assertEqual(result_2['dominant_emotion'], 'anger')

        # Test 3: disgust
        result_3 = emotion_detector('I feel disgusted just hearing about this')
        self.assertEqual(result_3['dominant_emotion'], 'disgust')

        # Test 4: sadness
        result_4 = emotion_detector('I am so sad about this')
        self.assertEqual(result_4['dominant_emotion'], 'sadness')

        # Test 5: fear
        result_5 = emotion_detector('I am really afraid that this will happen')
        self.assertEqual(result_5['dominant_emotion'], 'fear')


if __name__ == '__main__':
    unittest.main()
