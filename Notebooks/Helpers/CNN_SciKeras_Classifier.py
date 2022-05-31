from scikeras.wrappers import KerasClassifier
from sklearn.base import ClassifierMixin
from keras.callbacks import EarlyStopping
from .CNN_tensorflow_model import CNN_model
from tensorflow.keras.optimizers import Adadelta
import numpy as np


class CNN_classifier(KerasClassifier, ClassifierMixin):
    def __init__(
            self,
            model=CNN_model,
            random_state=42,
            verbose=1,
            batch_size=50,
            validation_split=0.1,
            epochs=40,
            optimizer=Adadelta(learning_rate=1),
            loss="binary_crossentropy",
            metrics=["accuracy"],
            **kwargs
    ):
        super(CNN_classifier, self).__init__(
            model=model,
            random_state=random_state,
            verbose=verbose,
            batch_size=batch_size,
            validation_split=validation_split,
            epochs=epochs,
            loss=loss,
            optimizer=optimizer,
            metrics=metrics,
            **kwargs
        )

    def fit(self, X, y, **fit_params):
        super().set_params(model__seq_len=X.shape[1])
        super().set_params(model__emb_dim=X.shape[2])
        return super().fit(X, y, **fit_params)

    def predict_proba(self, X):
        return super().predict_proba(X)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)
