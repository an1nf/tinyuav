# Loquercio et al. 2021 · *Learning high-speed flight in the wild* (Sci. Robot. 6:abg5810)

Qué se rescató y qué se descubrió al aplicarlo a nuestro stack. Todo en simulación (CPU); la
CNN usada es M int8 real salvo donde se indica GT (profundidad privilegiada, `UAV_FAKE_CNN=1`).

## Ideas del paper usadas
1. **Cota teórica de velocidad** (ec. 9–11, extendiendo a Falanga 2019), incluyendo la latencia
   de rotación.
2. **Experimento controlado del poste** (Fig. 5), corrido con profundidad perfecta y con la
   percepción real para separar fallos de percepción de fallos de planificación.
3. **Su tesis central:** los pipelines modulares acumulan latencia y conservadurismo porque "no
   modelan las interacciones entre módulos"; sus bases de comparación "adaptan su movimiento solo
   cuando están muy cerca" y "ordenan frenar".
4. **Cámara inclinada hacia arriba** en vuelo rápido.
5. **Sensibilidad a ruido de estimación y de empuje** (§S1, factor de empuje U(0.9, 1)).

## R4 · Cota teórica (`r4_speed_limit.py`)
Nano con alcance s = 3 m: el techo físico al esquivar es **7.3 m/s**. Nuestro stack, con sus
límites autoimpuestos, llega a 2.6 m/s si esquiva y a **1.8 m/s** con la ley de frenado
(`v_sense_cap`): usamos ~25 % de la envolvente.

Sensibilidad de v_max (a_lat = 2.2 m/s²):

| cambio | ganancia |
|---|---|
| alcance 3 → 4 m | **+1.43 m/s** |
| a_lat 2.2 → 4 m/s² | +0.88 m/s |
| 10 → 20 fps | +0.33 m/s |
| latencia 60 → 20 ms | +0.26 m/s |

→ Comprimir más la CNN es la palanca menos útil (coincide con Su et al. 2026). Lo que más paga
es el alcance fiable de percepción y permitir esquivar en vez de frenar.

## R5 · Experimento del poste (`r5_pole.py`, nano, 6 velocidades × 5)
Con la ley `brake` original, **el dron frenó por debajo de 0.2 m/s en el 100 % de las corridas y
empezó a esquivar a 0.3–1.0 m**, incluso con profundidad perfecta. Es el mismo modo de fallo que
Loquercio describe en sus bases de comparación. Con GT falló *más* que con la CNN (13/30 frente
a 17/30), lo que indicaba que el cuello de botella era la planificación, no la percepción.

Causas, instrumentadas con `planner.dbg`: **siete limitadores de velocidad multiplicativos en
serie** y tres interacciones entre módulos que no se modelaban.
1. El DP permitía pendiente 45° (SLOPE_MAX = 0.8 se cuantiza a 2 celdas/bin = 1.0) y su coste
   favorece el viraje más tardío posible. El limitador de FOV lleva la velocidad a 0 a 45°
   (FOV_LIM), así que cada esquive terminaba en frenada.
2. **Deadlock:** `backoff` ordena retroceder a 0.15 m/s y el limitador de FOV lo anula (retroceder
   es moverse a 180° de la cámara). Resultado: dron congelado ~9 s frente al poste.
3. El seguimiento usaba solo el primer bin del plan (0.2 m): el plan esquivaba a 1.4 m, pero el
   dron no se movía lateralmente hasta el último momento.
4. El paso lateral puro (pensado para salir de bloqueos) se usaba también al volver al carril, y
   "primero alinearse" anulaba el avance (timeouts tras pasar el obstáculo).

**Modo `CONSISTENT`** en `evasion.py` (desactivado por defecto; E3 no cambia):
- pendiente del plan ≤ tan(FOV_OK), con FOV_OK = 28°;
- objetivo lateral a v·0.6 s por delante;
- retroceso exento del FOV pero acotado a 0.6 m por bloqueo;
- paso lateral solo si no hay salida con pendiente factible.

| experimento | base | consistent | Fisher p |
|---|---|---|---|
| poste, GT | 13/30, 2 col | **30/30, 0 col** | 6e-7 |
| poste, CNN | 17/30, 6 col | **30/30, 0 col** | 5e-5 |
| corredores E3 nano 1.5 m/s | 12/20, 3 col, t 15.5 s | **20/20, 0 col, t 11.2 s** | 3e-3 |
| corredores E3 X3 1.5 m/s | **2/20** (18 timeouts) | **17/20**, holgura 0.69 m | 3e-6 |

El caso X3 explica los "timeouts por dron conservador" que v3 dejó pendientes: no era un problema
de sintonía, sino de módulos que se contradecían. Resultados re-verificados con el código final.

## R7 · Geometría cámara/ToF vs suelo (`r7_camera_tilt.py`)
A z = 0.65 m, con la cámara inclinada 8.6° hacia abajo:

| situación | zonas ToF que ven suelo a < 4 m | el eje toca el suelo a |
|---|---|---|
| hover | 50 % | — |
| acelerando con límites 'fast' (2.2 m/s², cabeceo 12.6°) | 75 % | 1.67 m |
| cámara +10° hacia arriba, en crucero | 38 % | — |
| cámara +20° hacia arriba, en crucero | 12 % | — |

Las zonas superiores siguen viendo obstáculos altos, pero la mitad del sensor más valioso (R3)
mira el piso. Cambiar la inclinación exige re-renderizar el dataset y reentrenar (el horizonte
se mueve en la imagen) y tocar el SDF.

## R8 · Ruido de estimación (`r8_state_noise.py`, nano 1 m/s, corredores, empuje U(0.9, 1))

| σ velocidad | consistent | brake |
|---|---|---|
| 0 | 10/10 | 8/10 |
| 0.05 | 10/10 | 9/10 |
| 0.10 | 7/10 | 7/10 |
| 0.20 | 6/10 | 6/10 |

- **Solo ruido blanco de 0.2 m/s: 10/10 (sin efecto).** **Solo sesgo de 0.2 m/s: 6/10.** El
  problema es la deriva (odometría), no el ruido.
- Olvidar el mapa más rápido (decay 0.95 o 0.98) **no** ayuda (5/10): no son celdas viejas.
- En los fallos, el dron pasa ~80 % del tiempo en BRAKE con v_lim = 0 (cree todo bloqueado).
- Durante esta prueba apareció un fallo propio: el retroceso sin cota retrocedió 5.7 m a ciegas.
  Quedó acotado a 0.6 m por bloqueo.
- **Requisito derivado:** sesgo del estimador de velocidad ≤ 0.05 m/s (flujo óptico + ToF de
  altura bien calibrado), o planificación egocéntrica/sin mapa. Mecanismo exacto: abierto.

## Pendiente / siguiente
- Hacer `CONSISTENT` el valor por defecto de la ley 'brake' y re-correr E4 completo con n ≥ 30.
- Subir el tope de velocidad: v_sense_cap usa A_BRAKE = 1.0 m/s², muy por debajo del físico
  (17 m/s² en el nano). Con el modo consistente, el siguiente paso es esquivar sin frenar.
- Diagnosticar el BRAKE persistente con deriva: posible desalineación entre la huella marcada
  como vista, el carril (coordenadas mundo) y el mapa.
- Idea de Loquercio para más adelante: aprendizaje privilegiado (profesor = este planificador
  con GT; alumno = red pequeña sobre la rejilla 12×12 CNN+ToF de R3) para reemplazar la cadena
  de limitadores por una política única.
