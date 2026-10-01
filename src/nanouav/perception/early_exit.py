"""
E2-D · Early-exit condicional: la CNN para en la capa más temprana
que da suficiente confianza, reduciendo latencia cuando el obstáculo
es obvio.

Concepto:
  En lugar de siempre ejecutar el decoder completo, extraemos
  una "señal de alarma" de la representación interna del encoder:
  si la activación media de la última capa del encoder supera un
  umbral, el obstáculo es lo suficientemente prominente como para
  tomar una decisión de evasión SIN completar el decoder.

  Esto no requiere modificar el modelo TFLite — se implementa como
  una política de dos pasos sobre la CNN existente:
    1. Correr solo el encoder (rápido) → estimar "peligro"
    2. Si peligro > umbral: retornar mapa grueso 12×12 escalado
       Si peligro ≤ umbral: correr decoder completo → mapa 96×96

Dos modos de implementación:
  A) Con modelo Keras (tu CPU): extracción real de activaciones
  B) Sin Keras (solo TFLite): proxy basado en flujo óptico + ToF
     como señal de early-exit (funciona en ESP32-S3)

Este módulo implementa ambos y los compara.
"""
import numpy as np
import os, sys, time

from nanouav.config import norm_to_m
from nanouav.sensors.optical_flow import DIV_THRESHOLD

DANGER_THRESHOLD = 0.65    # activación media encoder → "peligro detectado"
DEPTH_MIN, DEPTH_MAX = 0.2, 4.0


class EarlyExitCNN:
    """
    CNN con early-exit basado en activaciones del encoder.
    Requiere model.keras para extraer activaciones intermedias.
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

            # Modelo encoder: input → activación de la última capa encoder
            # En nuestra arquitectura: la capa re_lu_5 (12x12xF)
            enc_output = None
            for layer in full.layers:
                if 're_lu' in layer.name and enc_output is None:
                    pass  # primera relu
                if 're_lu' in layer.name:
                    enc_output = layer.output  # última relu del encoder

            if enc_output is None:
                enc_output = full.layers[len(full.layers)//2].output

            self._encoder_model = keras.Model(full.input, enc_output)
            self._full_model    = full
            self._loaded        = True
            print(f'[E2-D] Modelos cargados: encoder output {enc_output.shape}')
        except Exception as e:
            print(f'[E2-D] No se pudo cargar Keras: {e}')
            self._loaded = False

    @staticmethod
    def _dummy_loss(y_true, y_pred):
        import tensorflow as tf
        return tf.reduce_mean(tf.abs(y_true - y_pred))

    def danger_score(self, gray: np.ndarray) -> float:
        """
        Retorna score de peligro [0,1] basado en activaciones del encoder.
        Rápido: solo ejecuta la mitad del modelo.
        """
        self._load()
        if not self._loaded or self._encoder_model is None:
            return 0.5  # fallback neutro

        x = gray.astype(np.float32)[np.newaxis, :, :, np.newaxis] / 255.0
        enc = self._encoder_model(x, training=False).numpy()
        return float(np.mean(enc > 0.3))  # fracción de neuronas activas

    def predict(self, gray: np.ndarray, depth_gt: np.ndarray,
                variant: str = 'M') -> tuple:
        """
        Retorna (depth float32 96×96, exited_early bool, danger float).
        Si danger > DANGER_THRESHOLD: usa mapa grueso escalado (rápido).
        Si no: usa decoder completo (preciso).
        """
        danger = self.danger_score(gray)

        if danger > DANGER_THRESHOLD and self._loaded:
            # Early exit: upscale del mapa del encoder a 96×96
            x   = gray.astype(np.float32)[np.newaxis, :, :, np.newaxis] / 255.0
            enc = self._encoder_model(x, training=False).numpy()[0, :, :, 0]
            # Escalar 12×12 → 96×96 con interpolación bilineal simple
            import cv2
            coarse = cv2.resize(enc, (96, 96), interpolation=cv2.INTER_LINEAR)
            depth  = np.clip(norm_to_m(np.clip(coarse, 0, 1)), DEPTH_MIN, DEPTH_MAX).astype(np.float32)
            return depth, True, danger
        else:
            # Decoder completo
            from nanouav.perception.cnn import predict_depth
            depth = predict_depth(gray, depth_gt, variant)
            return depth, False, danger


class EarlyExitProxy:
    """
    Early-exit sin Keras: usa flujo óptico + ToF como proxy del encoder.
    Esta versión funciona en el ESP32-S3 con TFLite.

    Lógica:
      - Si div_óptico > div_threshold Y tof_min < tof_threshold:
          → Peligro obvio: usar ToF directamente, no ejecutar CNN
      - Si div_óptico > div_threshold Y tof_min >= tof_threshold:
          → Ejecutar CNN completa (peligro detectado por movimiento)
      - Si div_óptico <= div_threshold:
          → Sin peligro: no ejecutar CNN (idle)
    """

    def __init__(self, div_threshold=DIV_THRESHOLD, tof_threshold=0.8):
        self.div_threshold = div_threshold
        self.tof_threshold = tof_threshold

    def decide(self, divergence: float, tof_min: float) -> str:
        """
        Retorna: 'idle' | 'tof_only' | 'full_cnn'
        """
        if divergence <= self.div_threshold:
            return 'idle'
        if tof_min < self.tof_threshold:
            return 'tof_only'
        return 'full_cnn'

    def estimated_latency_ms(self, decision: str,
                             cnn_lat_ms: float = 0.638) -> float:
        """Latencia estimada por tipo de decisión."""
        return {'idle': 0.0, 'tof_only': 0.05, 'full_cnn': cnn_lat_ms}[decision]
