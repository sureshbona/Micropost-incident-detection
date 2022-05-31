from scikeras.wrappers import KerasClassifier
from sklearn.base import ClassifierMixin
from keras.callbacks import EarlyStopping
from .CBiGRU_tensorflow_model import CBiGRU_model


class CBiGRU_classifier(KerasClassifier, ClassifierMixin):
    def __init__(
            self,
            model=CBiGRU_model,
            random_state=42,
            verbose=1,
            batch_size=32,
            validation_split=0.1,
            epochs=5,
            optimizer="adam",
            loss="binary_crossentropy",
            callbacks=[
            EarlyStopping(
                monitor='val_accuracy',
                min_delta=0,
                patience=1,
                # verbose=0,
                mode='auto',
                baseline=None,
                restore_best_weights=True)
            ],
            metrics=["accuracy"],
            **kwargs
    ):
        super(CBiGRU_classifier, self).__init__(
            model=model,
            random_state=random_state,
            verbose=verbose,
            batch_size=batch_size,
            validation_split=validation_split,
            epochs=epochs,
            callbacks=callbacks,
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            **kwargs
        )

    def fit(self, X, y, **fit_params):
        super().set_params(model__emb_dim=X.shape[2])
        return super().fit(X, y, **fit_params)

    def predict_proba(self, X):
        return super().predict_proba(X)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)
