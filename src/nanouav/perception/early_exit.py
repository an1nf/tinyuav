"""
E2-D · Conditional early exit: the CNN stops at the earliest layer
that gives enough confidence, reducing latency when the obstacle
is obvious.

Concept:
  Instead of always running the full decoder, we extract an
  "alarm signal" from the encoder's internal representation:
  if the mean activation of the last encoder layer exceeds a
  threshold, the obstacle is prominent enough to make an
  avoidance decision WITHOUT completing the decoder.

  This does not require modifying the TFLite model — it is implemented
  as a two-step policy on top of the existing CNN:
    1. Run only the encoder (fast) → estimate "danger"
    2. If danger > threshold: return the upscaled 12×12 coarse map
       If danger ≤ threshold: run the full decoder → 96×96 map

Two implementation modes:
  A) With the Keras model (your CPU): real activation extraction
  B) Without Keras (TFLite only): proxy based on optical flow + ToF
     as the early-exit signal (works on the ESP32-S3)

This module implements both and compares them.
"""
import numpy as np
import os, sys, time

from nanouav.config import norm_to_m
from nanouav.sensors.optical_flow import DIV_THRESHOLD

DANGER_THRESHOLD = 0.65    # mean encoder activation → "danger detected"
DEPTH_MIN, DEPTH_MAX = 0.2, 4.0


class EarlyExitCNN:
    """
    CNN with early exit based on encoder activations.
    Requires model.keras to extract intermediate activations.
    """

    def __init__(self, keras_path: str):
        self._full_model   = None
        self._encoder_model = None
        self._keras_path   = keras_path
        self._loaded       = False

    def _load(self):
        if self._loaded:
            return
        try:
            import keras
            from keras import backend as K
            import tensorflow as tf

            full = keras.models.load_model(
                self._keras_path,
                custom_objects={'depth_loss': self._dummy_loss})

            # Encoder model: input → activation of the last encoder layer
            # In our architecture: the re_lu_5 layer (12x12xF)
            enc_output = None
            for layer in full.layers:
                if 're_lu' in layer.name and enc_output is None:
                    pass  # first relu
                if 're_lu' in layer.name:
                    enc_output = layer.output  # last relu of the encoder

            if enc_output is None:
                enc_output = full.layers[len(full.layers)//2].output

            self._encoder_model = keras.Model(full.input, enc_output)
            self._full_model    = full
            self._loaded        = True
            print(f'[E2-D] Models loaded: encoder output {enc_output.shape}')
        except Exception as e:
            print(f'[E2-D] Could not load Keras: {e}')
            self._loaded = False

    @staticmethod
    def _dummy_loss(y_true, y_pred):
        import tensorflow as tf
        return tf.reduce_mean(tf.abs(y_true - y_pred))

    def danger_score(self, gray: np.ndarray) -> float:
        """
        Return a danger score [0,1] based on encoder activations.
        Fast: only runs half of the model.
        """
        self._load()
        if not self._loaded or self._encoder_model is None:
            return 0.5  # neutral fallback

        x = gray.astype(np.float32)[np.newaxis, :, :, np.newaxis] / 255.0
        enc = self._encoder_model(x, training=False).numpy()
        return float(np.mean(enc > 0.3))  # fraction of active neurons

    def predict(self, gray: np.ndarray, depth_gt: np.ndarray,
                variant: str = 'M') -> tuple:
        """
        Return (depth float32 96×96, exited_early bool, danger float).
        If danger > DANGER_THRESHOLD: use the upscaled coarse map (fast).
        Otherwise: use the full decoder (accurate).
        """
        danger = self.danger_score(gray)

        if danger > DANGER_THRESHOLD and self._loaded:
            # Early exit: upscale the encoder map to 96×96
            x   = gray.astype(np.float32)[np.newaxis, :, :, np.newaxis] / 255.0
            enc = self._encoder_model(x, training=False).numpy()[0, :, :, 0]
            # Scale 12×12 → 96×96 with simple bilinear interpolation
            import cv2
            coarse = cv2.resize(enc, (96, 96), interpolation=cv2.INTER_LINEAR)
            depth  = np.clip(norm_to_m(np.clip(coarse, 0, 1)), DEPTH_MIN, DEPTH_MAX).astype(np.float32)
            return depth, True, danger
        else:
            # Full decoder
            from nanouav.perception.cnn import predict_depth
            depth = predict_depth(gray, depth_gt, variant)
            return depth, False, danger


class EarlyExitProxy:
    """
    Early exit without Keras: uses optical flow + ToF as a proxy for the encoder.
    This version works on the ESP32-S3 with TFLite.

    Logic:
      - If optical_div > div_threshold AND tof_min < tof_threshold:
          → Obvious danger: use ToF directly, do not run the CNN
      - If optical_div > div_threshold AND tof_min >= tof_threshold:
          → Run the full CNN (danger detected from motion)
      - If optical_div <= div_threshold:
          → No danger: do not run the CNN (idle)
    """

    def __init__(self, div_threshold=DIV_THRESHOLD, tof_threshold=0.8):
        self.div_threshold = div_threshold
        self.tof_threshold = tof_threshold

    def decide(self, divergence: float, tof_min: float) -> str:
        """
        Returns: 'idle' | 'tof_only' | 'full_cnn'
        """
        if divergence <= self.div_threshold:
            return 'idle'
        if tof_min < self.tof_threshold:
            return 'tof_only'
        return 'full_cnn'

    def estimated_latency_ms(self, decision: str,
                             cnn_lat_ms: float = 0.638) -> float:
        """Estimated latency per decision type."""
        return {'idle': 0.0, 'tof_only': 0.05, 'full_cnn': cnn_lat_ms}[decision]
