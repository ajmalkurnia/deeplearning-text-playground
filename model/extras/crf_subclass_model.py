"""
Wrapping model that has CRF layer to compute crf_loss
source @jaspersjsun https://github.com/tensorflow/addons/pull/1999/files
"""
import tensorflow as tf
from tensorflow_addons.text.crf import crf_log_likelihood


def unpack_data(data):
    if len(data) == 2:
        return data[0], data[1], None
    elif len(data) == 3:
        return data
    else:
        raise TypeError("Expected data to be a tuple of size 2 or 3.")


class ModelWithCRFLoss(tf.keras.Model):
    """Wrapper around the base model for custom training logic."""

    def __init__(self, base_model, **kwargs):
        super(ModelWithCRFLoss, self).__init__(**kwargs)
        self.base_model = base_model

    def call(self, inputs):
        return self.base_model(inputs)

    def compute_loss(self, x, y, sample_weight, training=False):
        y_pred = self(x, training=training)
        _, potentials, sequence_length, chain_kernel = y_pred
        crf_loss = -crf_log_likelihood(potentials,
                                       y, sequence_length, chain_kernel)[0]
        internal_loss = self.compiled_loss(
            y, potentials, regularization_losses=self.losses)
        if sample_weight is not None:
            crf_loss = crf_loss * sample_weight

        return tf.reduce_mean(crf_loss), internal_loss

    def train_step(self, data):
        x, y, sample_weight = unpack_data(data)

        with tf.GradientTape() as tape:
            crf_loss, internal_losses = self.compute_loss(
                x, y, sample_weight, training=True
            )
            total_loss = crf_loss + \
                tf.cast(tf.reduce_sum(self.losses), crf_loss.dtype)

        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.trainable_variables))

        return {"loss": crf_loss, "internal_losses": internal_losses}

    def test_step(self, data):
        x, y, sample_weight = unpack_data(data)
        crf_loss, internal_losses = self.compute_loss(x, y, sample_weight)
        return {"loss": crf_loss, "internal_losses": internal_losses}
