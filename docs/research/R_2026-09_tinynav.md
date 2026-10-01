# TinyNav · Roy et al., arXiv 2603.11071 (mar. 2026)

Robot terrestre (tanque) con ESP32-P4 y una cámara ToF MaixSense A010 (100×100 → 25×25 por
binning; 70°×60°; 0.2–2.5 m; 20 fps; UART). CNN 2D de 23 k parámetros sobre **20 frames de
profundidad apilados como canales** (TFLM no soporta LSTM/GRU/atención), entrenada por imitación
de un piloto humano. Latencia 30 ms, int8 conserva ~99.8 %. Núcleos separados para el control
y la inferencia, con una ventana deslizante en memoria compartida.

**Valoración crítica:** la correlación con el piloto es ~0.6. La única métrica en lazo cerrado
es "40 vueltas sin choque" en pistas parecidas a las de entrenamiento; en pistas nuevas hay
contactos. No hay línea base. Lo útil es la ingeniería de despliegue, no el resultado.

## Qué se rescató
1. **"Hasta ~50 k parámetros a 20 fps en ESP32"**: consistente con nuestra M (26 k). La P4
   (768 KB de L2 y 32 MB de PSRAM según el paper; ISP y MIPI-CSI) permitiría la variante L o
   160×120 sin el límite de arena de 180 KB del S3. Por R4, más cómputo casi no sube la
   velocidad con obstáculos estáticos; por R10 sí ayuda con los dinámicos (latencia).
2. **Sensor A010 frente al nuestro** (cota R4, nano, límites del stack):

   | sensor | alcance | HFOV | v_max esquivando | v_max frenando | cruce invisible si v_obj > k·v_dron |
   |---|---|---|---|---|---|
   | CNN mono | 3.0 m | 60° | 2.64 | 1.82 | k = 0.58 |
   | VL53L5CX | 3.7 m | 40° | 3.25 | 2.09 | **k = 0.36** |
   | A010 | 2.5 m | 70° | 2.20 | 1.60 | k = 0.70 |

   Gana en ancho y pierde en alcance, que es la palanca principal (R4). Nuevo para R10: la
   detección dinámica basada en el ToF de 40° tiene un cono ciego aún mayor (k = 0.36).
3. **Frames apilados → velocidad** (R11, `r11_vel_net.py`). Con 5.8 k secuencias (4 frames a
   10 Hz, giróscopo y altura como entradas), CNNs tiny solo-ToF (21 k), solo-cámara (51 k) y
   ambas (70 k) quedan **casi en la línea base** (RMSE 0.43–0.58 m/s frente a desviación de las
   etiquetas 0.66/0.58): no aprenden flujo desde cero a esta escala. **Resultado negativo.**
4. **"Flow deck frontal" geométrico** (R11b, `r11b_flow_deck.py`). Idea propia a partir de R7: la
   mitad inferior de la imagen ve el suelo. Farnebäck + proyección al plano del suelo (altura y
   actitud conocidas) + búsqueda de (vx, vy) con compensación de giro, sin aprendizaje.
   - Mediana del error: **0.12 / 0.09 m/s**, unas 3× mejor que la red aprendida.
   - Cola pesada: p90 de 1.3 / 1.0 m/s. Solo el 18 % de los casos cumple ≤ 0.05 m/s en ambos
     ejes y el 37 % ≤ 0.10.
   - Fallos: suelo sin textura (79 % de fallos), velocidad > 1.2 m/s a 10 Hz (47 %) y,
     probablemente, texturas periódicas tipo tablero (aliasing del flujo; no verificado).
   - Conclusión: la física supera al aprendizaje con pocos datos, pero no alcanza el requisito
     de R8 a 10 Hz. Pasos siguientes: 30 fps, un filtro que fusione la IMU y rechazar
     estimaciones con baja confianza (textura pobre).
   - Recomendación práctica para el nano: un flow deck descendente dedicado (PMW3901 + ToF de
     altura, del orden de 1–2 g; verificar peso) es la vía estándar. La versión frontal queda
     como redundancia sin sensor extra.

## Pendiente
- R11b a 30 fps y con etiqueta del tipo de suelo; integrar en el lazo con R8 (sesgo 0.2).
- Simular la A010 como sensor alternativo en E3/R10 (70°, 2.5 m, 25×25).
