from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from six.moves.urllib.request import urlopen

import numpy as np
import tensorflow as tf

# Data sets
IRIS_TRAINING = "dumb_data/data_training.csv"
IRIS_TEST = "dumb_data/data_test.csv"

def main():
    # If the training and test sets aren't stored locally, download them.
    if not os.path.exists(IRIS_TRAINING):
        raw = urlopen(IRIS_TRAINING_URL).read()
        with open(IRIS_TRAINING, "wb") as f:
            f.write(raw)

    if not os.path.exists(IRIS_TEST):
        raw = urlopen(IRIS_TEST_URL).read()
        with open(IRIS_TEST, "wb") as f:
            f.write(raw)


    # Load datasets.
    training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
        filename=IRIS_TRAINING,
        target_dtype=np.int,
        features_dtype=np.float32)

    test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
        filename=IRIS_TEST,
        target_dtype=np.int,
        features_dtype=np.float32)


    # Specify that all features have real-value data
    # tf.feature_column.numeric_column used for continuous feature data
    feature_columns = [tf.feature_column.numeric_column("x", shape=[3])]

    # Build 3 layer Deep Neural Network (DNN) with 10, 20, 10 units respectively.
    classifier = tf.estimator.DNNClassifier(
        feature_columns=feature_columns,
        hidden_units=[10, 20, 10],
        n_classes=3,
        model_dir="./iris_model")

    # Define the training inputs
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": np.array(training_set.data)},
        y=np.array(training_set.target),
        num_epochs=None,
        shuffle=True)

    # Train model.
    classifier.train(input_fn=train_input_fn, steps=2000)

    # Define the test inputs
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": np.array(test_set.data)},
        y=np.array(test_set.target),
        num_epochs=1,
        shuffle=False)

    # Evaluate accuracy.
    accuracy_score = classifier.evaluate(input_fn=test_input_fn)["accuracy"]

    print("\nTest Accuracy: {0:f}\n".format(accuracy_score))


# def predict():
    # Classify two new flower samples.
    new_samples = np.array(
        [[1, 5, 4.5],
         [2, 12, 5.0]], dtype=np.float32)
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": new_samples},
        num_epochs=1,
        shuffle=False)

    predictions = list(classifier.predict(input_fn=predict_input_fn))
    predicted_classes = [p["classes"] for p in predictions]

    print("New Samples, Class Predictions:{}\n".format(predicted_classes))

if __name__ == '__main__':
    main()
    # predict()
