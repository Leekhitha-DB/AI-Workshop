import os
import tempfile
import unittest

from train_model import load_dataset, predict_emotion, train_text_classifier


class TrainModelTests(unittest.TestCase):
    def test_load_dataset(self):
        sample_data = """i feel sad;sadness
i am happy;joy
i am angry;anger
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as tmp:
            tmp.write(sample_data)
            tmp_path = tmp.name

        try:
            df = load_dataset(tmp_path)
            self.assertEqual(df.shape, (3, 2))
            self.assertListEqual(df["label"].tolist(), ["sadness", "joy", "anger"])
            self.assertListEqual(df["text"].tolist(), ["i feel sad", "i am happy", "i am angry"])
        finally:
            os.remove(tmp_path)

    def test_train_text_classifier(self):
        sample_data = """i feel sad;sadness
i feel terrible;sadness
i am so happy;joy
i feel joyful;joy
i am angry;anger
i feel furious;anger
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "train.txt")
            model_path = os.path.join(tmpdir, "model.joblib")
            vectorizer_path = os.path.join(tmpdir, "vectorizer.joblib")

            with open(data_path, "w", encoding="utf-8") as f:
                f.write(sample_data)

            model, vectorizer = train_text_classifier(data_path, model_path, vectorizer_path)
            self.assertTrue(os.path.exists(model_path))
            self.assertTrue(os.path.exists(vectorizer_path))

            prediction = predict_emotion("i am feeling sad and hopeless", model, vectorizer)
            self.assertIn(prediction, ["sadness", "joy", "anger"])


if __name__ == "__main__":
    unittest.main()
