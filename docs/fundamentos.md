# Fundamentos de nanouav

*2026-10-15 · nanouav 0.24.0.* Este documento explica, en un solo lugar, **qué** se ha construido, **por
qué** funciona así (las razones matemáticas) y **cómo** está organizado el proyecto. Los informes de cada
iteración (`docs/research/R*.md`) tienen los detalles y los números de cada experimento; aquí se reúne lo
que los sostiene.

Contenido:

1. [El problema](#1-el-problema)
2. [Historia del proyecto en una tabla](#2-historia-del-proyecto)
3. [El nano-UAV: cámara monocular + ToF](#3-el-nano-uav-cámara-monocular--tof)
4. [La recreación de Loquercio et al. 2021 (R18)](#4-la-recreación-de-loquercio-et-al-2021-r18)
5. [Simuladores: el propio, Gazebo y Flightmare](#5-simuladores)
6. [Infraestructura del proyecto](#6-infraestructura-del-proyecto)
7. [Estado actual y lo que sigue](#7-estado-actual-y-lo-que-sigue)
8. [Referencias](#8-referencias)

---

## 1. El problema

Un dron debe volar de un punto de partida a una meta (40 m adelante) por un entorno con obstáculos que
**no conoce de antemano**, usando solo lo que ven sus sensores a bordo y su propia estimación de estado.
No hay mapa previo, ni GPS, ni cómputo externo.

Hay dos escalas de plataforma en el proyecto:

| | nano-UAV (objetivo original) | dron ágil (referencia: Loquercio 2021) |
|---|---|---|
| sensores | cámara monocular 96 × 96 en grises + ToF 8 × 8 (VL53L5CX) | cámara estéreo (RealSense D435, base 0.1 m) |
| cómputo | microcontrolador (ESP32-S3), red int8 | Jetson TX2 |
| velocidad | ~1–2 m/s | 3–10 m/s |
| decisión | planificador local sobre la profundidad estimada | red que predice trayectorias directamente |

La pregunta de fondo es cómo pasar de **percepción local y ruidosa** a **movimiento seguro**. Hay dos
familias de respuesta:

- **Modular:** percibir → mapear → planificar → controlar. Es interpretable, pero cada etapa añade
  latencia y error, y a velocidad alta los errores se acumulan.
- **Aprendida de extremo a extremo** (Loquercio): una red pasa de la percepción a trayectorias, entrenada
  en simulación por imitación de un experto que sí conoce el mundo.

El proyecto recorrió la primera (E0–R17) y después recreó la segunda (R18–R22) para poder compararlas y,
eventualmente, llevar lo aprendido al nano-UAV.

---

## 2. Historia del proyecto

| etapa | qué se hizo | resultado que importa |
|---|---|---|
| E0–E4 | plan de evaluación del nano: latencia, energía, calibración ToF, lazo cerrado, misiones | pipeline completo en el simulador propio |
| R1–R3 | redes de profundidad diminutas (LNCA), seguimiento de objetivos | huella int8 y latencia en el ESP32-S3 |
| R4–R8 | límite de velocidad por alcance de percepción, ley consistente, inclinación de cámara, ruido de estado | la velocidad la limita el alcance del sensor, no el control |
| R9–R11 | obstáculos dinámicos, deriva, velocidad desde la cámara | capas opcionales del planificador |
| R12–R13 | planificador tipo M-H (Loquercio) sobre la rejilla local, pasos estrechos | `MHPlanner` |
| R14–R17 | métricas por clase, falsas detecciones, entrenamiento completo, troncos finos | límites del sensor monocular |
| **R18** | recreación completa de Loquercio 2021: estéreo + SGM, experto M-H privilegiado, estudiante, DAgger | 22–23/40 en el bosque de prueba (experto 40/40) |
| R19–R20 | seguir una persona / un globo con la política de R18 | misiones sobre el mismo estudiante |
| R21 | el estudiante en Gazebo (ROS 2) | nodo y exportación a TFLite |
| **R22** | el estudiante en Flightmare (el simulador del paper) y entrenamiento dentro de él | 69/90 en el almacén: elegir con su propio SGM (`sgm`) y no soltar un plan libre (`--keep`) |
| R23 | por qué el estudiante se estanca: etiquetas que siguen una ruta que no ve | `--label-ref ctg` (aprende mejor, no vuela mejor: sin confirmar) |

---

## 3. El nano-UAV: cámara monocular + ToF

### 3.1 Por qué hace falta el ToF

Una sola imagen no determina la escala. Si la escena se agranda por un factor $s$ y se aleja por el
mismo factor, la imagen es idéntica. Una red de profundidad monocular aprende una escala promedio del
entrenamiento, y en un entorno distinto se equivoca por un factor casi constante:

$$\hat Z_\text{CNN}(u,v) \approx \frac{1}{s}\, Z(u,v).$$

El ToF 8 × 8 mide distancias métricas en 64 zonas. Si en cada zona válida $i$ se compara la medida del ToF
con la profundidad de la red agregada a esa zona, se obtiene una estimación de la escala:

$$\hat s = \operatorname{mediana}_i \frac{Z^\text{ToF}_i}{\hat Z^\text{CNN}_i}.$$

Se usa la **mediana**, no la media, porque una zona con un borde de objeto o un reflejo da un cociente
disparatado. La estimación se rechaza si hay pocas zonas válidas o si la dispersión relativa (MAD) es
alta. Como la escala es multiplicativa, se suaviza en logaritmo:
$\log s_t = (1-\alpha)\log s_{t-1} + \alpha \log \hat s$, con $\alpha = 0.15$
(`perception/tof_calibrator.py`).

### 3.2 Fusión

La red da **forma** (dónde están los bordes, 96 × 96 píxeles) y el ToF da **escala y cercanía**
(64 zonas, métricas). `fuse_tof` corrige la profundidad de la red en las zonas donde el ToF ve algo más
cerca de lo que la red estima, para que un obstáculo cercano nunca se subestime.

### 3.3 El límite de velocidad lo pone el alcance del sensor (R4)

Para frenar ante un obstáculo detectado a distancia $d$, con latencia $t_r$ y desaceleración máxima
$a$:

$$v\,t_r + \frac{v^2}{2a} \le d \quad\Longrightarrow\quad v_\text{max} = a\left(\sqrt{t_r^2 + \frac{2d}{a}} - t_r\right).$$

Con un alcance de percepción de pocos metros, $v_\text{max}$ queda en pocos m/s aunque el vehículo pueda
acelerar mucho más. Por eso la velocidad del nano está limitada por el sensor y no por el control, y por
eso el paper usa estéreo, que alcanza unos 10 m.

### 3.4 Planificadores locales

- `EvasionPlanner`: rejilla local 2D de 0.1 m alimentada por la profundidad fusionada. Hace programación
  dinámica lateral sobre un carril de ±2.3 m y aplica un filtro de seguridad con la ecuación de frenado
  anterior.
- `MHPlanner` (R12–R13): sobre la misma rejilla calcula un campo de costo hasta la meta con Dijkstra y
  evalúa 180 arcos como hipótesis. Es una versión clásica, sin aprendizaje, de la idea de Loquercio.

---

## 4. La recreación de Loquercio et al. 2021 (R18)

### 4.1 Aprendizaje privilegiado: la idea central

Entrenar una red para que vuele directamente por refuerzo es lento e inestable. Loquercio separa el
problema en dos:

1. Un **experto privilegiado** que conoce el mapa completo (la nube de puntos del entorno) y el estado
   exacto. Con eso puede calcular trayectorias libres de choque por búsqueda, sin aprender nada.
2. Un **estudiante** que solo ve lo que verá el dron real (profundidad estéreo, velocidad, orientación,
   dirección a la meta) y aprende a **imitar** al experto.

Es la idea de *learning by cheating* (Chen et al. 2019). El experto «hace trampa» con el mapa durante el
entrenamiento, y el estudiante aprende a obtener el mismo resultado sin la trampa. **El mapa nunca entra
en la red.** Solo se usa para etiquetar (el experto) y para decidir si hubo choque (el juez de la
simulación).

### 4.2 Estéreo y SGM

Dos cámaras separadas una base $B$ ven un punto a profundidad $Z$ desplazado $d$ píxeles entre imágenes
(disparidad). Con focal $f$ en píxeles:

$$Z = \frac{f\,B}{d}, \qquad \delta Z \approx \frac{Z^2}{f\,B}\,\delta d.$$

El error crece con el **cuadrado** de la distancia. Con $B = 0.1$ m y un error de disparidad
$\delta d = 0.25$ px, a 10 m:

| resolución | $f$ (HFOV 90°) | $\delta Z$ a 10 m |
|---|---|---|
| 320 × 240 (simulador propio) | 160 px | 1.56 m |
| 640 × 480 (VGA, el paper; Flightmare desde 0.20.0) | 320 px | 0.78 m |

Por eso el paper usa VGA, y por eso la profundidad útil se corta en 10 m (`StereoRig.max_range`).

La disparidad se calcula con **Semi-Global Matching** (Hirschmüller 2008). SGM minimiza, sobre el mapa de
disparidades $D$:

$$E(D) = \sum_p C(p, D_p) + \sum_{q\in N_p} P_1\,[|D_p - D_q| = 1] + \sum_{q\in N_p} P_2\,[|D_p - D_q| > 1].$$

El primer término compara las imágenes. Los otros dos penalizan saltos pequeños ($P_1$) y grandes ($P_2$)
entre vecinos, y así favorecen superficies suaves con bordes nítidos. Se optimiza por programación
dinámica en varias direcciones. El proyecto usa `cv2.StereoSGBM` (modo 3WAY, bloque 5, $P_1 = 8b^2$,
$P_2 = 32b^2$, unicidad 10, filtro de motas), más un filtro de textura: donde la desviación local es menor
que 4 niveles de gris, el píxel queda inválido, como en la RealSense.

**¿Por qué no darle a la red la profundidad perfecta?** Porque en el dron real solo habrá SGM, con sus
huecos y errores. El paper muestra que entrenar con profundidad perfecta o con RGB transfiere peor que
entrenar con SGM simulado (Fig. 7a). La profundidad es una **abstracción** de la imagen: quita la textura
y el color, que cambian mucho entre simulación y realidad, y deja la geometría, que no cambia. Esa
abstracción es la clave de que lo entrenado en simulación sirva en el mundo real.

### 4.3 Los entornos: bosques de Poisson

Los árboles de prueba se colocan con un proceso de Poisson de densidad $\delta$ (árboles/m²): el número
de árboles en un área $A$ es $N \sim \text{Poisson}(\delta A)$ y sus posiciones son uniformes. Un vuelo
recto de longitud $L$ con un dron de radio $r_q$ choca si algún tronco de radio $r$ cae en una franja de
ancho $2(r + r_q)$. La probabilidad de que la franja esté vacía es

$$P(\text{línea recta libre}) = e^{-\delta\, L\, 2(r + r_q)}.$$

Con la prueba del paper ($\delta = 1/25$, $L = 40$ m, $r = 0.3$, $r_q = 0.2$):
$P = e^{-1.6} \approx 0.20$. **El 80 % de los vuelos en línea recta chocarían**, así que la prueba exige
esquivar de verdad. Por eso un piloto ciego, que va recto a la meta, sirve de referencia inferior.

### 4.4 El experto privilegiado

**Referencia global $\tau_\text{gbl}$.** El experto calcula una ruta 2D libre de choques en una rejilla
(0.2 m) con Dijkstra, sobre el campo de distancias a los obstáculos con un inflado de $r_q + 0.15$ m. El
costo crece cerca de los obstáculos, se recortan los atajos y se suaviza con Chaikin
(`expert.global_plan`). Recorrida a $v_\text{des}$, da puntos de referencia $\tau_\text{ref}$ para el
siguiente segundo.

**Costo de una trayectoria** (Eq. 3–4 del paper). Para una trayectoria $\tau$ de 1 s:

$$C(\tau) = \int_0^1 \lambda_c\, C_\text{col}(\tau(t))\,dt + \int_0^1 (\tau(t) - \tau_\text{ref}(t))^\top Q\,(\tau(t) - \tau_\text{ref}(t))\,dt,$$

$$C_\text{col}(d) = \begin{cases} 4 - d^2/r_q^2 & d \le 2r_q\\ 0 & d > 2r_q\end{cases}$$

donde $d$ es la distancia del centro del dron al obstáculo más cercano. $C_\text{col}$ vale 4 al tocar
($d = 0$) y 0 a partir de dos radios, y es continua en $d = 2r_q$. Valores usados: $r_q = 0.2$ m,
$\lambda_c = 1000$, $Q = \operatorname{diag}(10, 10, 20)$ (desviarse en altura cuesta el doble) y
$dt = 0.1$ s.

**Muestreo de Metropolis-Hastings.** El problema es multimodal: un árbol puede esquivarse por la
izquierda o por la derecha, y ambas son buenas. En vez de una sola trayectoria óptima, el experto
muestrea la distribución

$$p(\tau) \propto e^{-C(\tau)}.$$

Cada trayectoria es una B-spline cúbica sujeta (5 puntos de control, nudos $[0,0,0,0,0.5,1,1,1,1]$). Los
dos primeros puntos de control los fijan la posición y la velocidad actuales, así que se muestrean los
tres últimos, en coordenadas esféricas alrededor del dron. Una cadena propone
$\theta' = \theta + \sigma\,\epsilon$ y la acepta con probabilidad

$$\alpha = \min\left(1,\; \frac{p(\tau')}{p(\tau)}\right) = \min\left(1,\; e^{-(C(\tau') - C(\tau))}\right).$$

Corren 100 cadenas, 50 000 muestras en total, con varianzas crecientes (×2, ×5, ×10) para explorar
primero cerca y luego lejos. De las muestras libres (distancia mínima mayor que $r_q$) se toman las
**$k = 3$ mejores que difieren entre sí** más de 0.3 m en promedio: esos son los modos, las etiquetas del
estudiante (`expert.Expert.label`).

### 4.5 El estudiante

**Entrada:**
- profundidad SGM de 120 × 160, normalizada $z \mapsto 2\min(z/10, 1) - 1$, con los huecos en −1;
- estado de 15 números: velocidad en el marco de rumbo (/10), matriz de rotación (9) y la dirección
  unitaria al punto 1 s adelante en la línea a la meta (3).

**Ojo (R23):** las etiquetas siguen $\tau_\text{gbl}$ y esa dirección apunta a la recta, así que las
etiquetas no son función de lo que ve el estudiante. Eso fija un piso de R-WTA ~45, el valor en que se
estanca su validación, y le enseña a no fiarse de la dirección. La corrección (`--label-ref ctg`) replanea
la referencia de las etiquetas desde la posición actual por el costo hasta la meta, pegado a la recta.
Ver [R23](research/R23_direccion_y_etiquetas.md).

**Red:** MobileNet-V3 Small sobre la profundidad, una rama densa para el estado y cabezas que producen
**$M = 3$ hipótesis**. Cada hipótesis tiene 10 puntos a $t = 0.1 \ldots 1$ s y un costo de choque
predicho (softplus).

**Por qué varias hipótesis y no una.** Si se entrena una sola salida con error cuadrático, el óptimo es
la **media condicional** de las etiquetas:
$\arg\min_{\hat\tau} \mathbb{E}\,\|\tau - \hat\tau\|^2 = \mathbb{E}[\tau \mid x]$. Ante un árbol de frente,
con la mitad de las etiquetas a la izquierda y la mitad a la derecha, la media va **recto contra el
árbol**. Con varias salidas y una pérdida que asigna cada etiqueta a la salida más cercana, cada salida se
especializa en un modo.

**Pérdida R-WTA** (*relaxed winner-takes-all*, Eq. 5). Sea $e_k$ la etiqueta $k$ y $n_m$ la predicción
$m$:

$$L_\text{R-WTA} = \sum_k \sum_m \alpha_{km}\,\|e_k - n_m\|^2, \qquad \alpha_{km} = \begin{cases}1-\varepsilon & m = \arg\min_{m'} \|e_k - n_{m'}\|\\ \varepsilon/(M-1) & \text{si no}\end{cases}$$

con $\varepsilon = 0.05$. El pequeño peso para las perdedoras evita que una hipótesis quede «muerta» sin
recibir nunca gradiente.

**Cabeza de costo.** Para cada hipótesis predicha se calcula su costo de choque verdadero con un mapa
local de distancias guardado con cada muestra (ESDF 2D de 16 × 16 m a 0.25 m, sin suelo, más la altura
sobre el suelo), y la red aprende a predecirlo:

$$L = \lambda_1 L_\text{R-WTA} + \lambda_2 \sum_m (c_m - C_\text{col}(n_m))^2, \qquad \lambda_1 = 10,\ \lambda_2 = 0.1.$$

En vuelo, ese costo predicho sirve para elegir entre las hipótesis sin conocer el mapa.

**Cabeza espacial (0.18.1).** La versión original promediaba el mapa de rasgos de la red
(*global average pooling*). Ese promedio es invariante a la posición: un tronco a la izquierda y uno a la
derecha dan el mismo vector, pero de eso depende el lado por el que esquivar. La cabeza espacial
(convolución 1 × 1 y aplanado) conserva *dónde* está cada cosa. En nuestro simulador mejoró de 3 a 7 m/s
(22/30, el mejor resultado en ese rango) y empeoró a 10 m/s.

### 4.6 De la predicción al vuelo

**Proyección sobre un polinomio** (Eq. 8). Los 10 puntos predichos no forman una trayectoria volable:
no empiezan necesariamente en la velocidad y la aceleración actuales. Por eje se ajusta un polinomio de
grado 5 $\mu(t) = \sum_{j=0}^5 c_j t^j$ por mínimos cuadrados, **con restricciones de igualdad**:

$$\min_c \sum_{i=1}^{10} \|\mu(t_i) - p_i\|^2 \quad \text{s.a.}\quad \mu(0) = p,\ \dot\mu(0) = v,\ \ddot\mu(0) = a.$$

Con la matriz de base $A$ ($A_{ij} = t_i^j$) y las restricciones $C_\text{eq}\,c = b$, el óptimo resuelve
el sistema KKT

$$\begin{bmatrix} 2A^\top A & C_\text{eq}^\top \\ C_\text{eq} & 0 \end{bmatrix} \begin{bmatrix} c \\ \lambda \end{bmatrix} = \begin{bmatrix} 2A^\top p \\ b \end{bmatrix},$$

cuya matriz se invierte una sola vez (`trajectory.poly5_fit`). Así cada plan nuevo empalma con el estado
actual sin saltos de velocidad ni de aceleración.

**Escala de tiempo $\beta$.** El paper fija la velocidad media del plan a $v_\text{des}$ reproduciendo el
polinomio a otro ritmo, $\tau = \beta (t - t_0)$:

$$p(t) = \mu(\tau), \quad \dot p = \beta\,\mu'(\tau), \quad \ddot p = \beta^2 \mu''(\tau), \qquad \beta = \frac{v_\text{des}}{\|p_{10} - p\| / 1\,\text{s}}.$$

Sutileza (corregida en 0.17.0): si el polinomio se ajusta con $\dot\mu(0) = v$ y luego se escala, la
velocidad comandada al inicio es $\beta v$, no $v$. A 3 m/s la red predecía unos 4.3 m por segundo
($\beta \approx 0.7$), es decir, una caída del 30 % de velocidad 15 veces por segundo. La versión
`scaled` ajusta con $\dot\mu(0) = v/\beta$ y $\ddot\mu(0) = a/\beta^2$, de modo que tras escalar
$\dot p(t_0) = v$ y $\ddot p(t_0) = a$ exactos.

**Elección de la hipótesis (el paper, `--select snap`).** Entre las $M$, se consideran las de costo
predicho cercano al mínimo ($c^*/c_k \ge 0.95$) y de ellas se toma la de menor **snap integrado**
$\int_0^1 \|\mu^{(4)}\|^2 dt$, la más suave según Mellinger y Kumar. Para un polinomio de grado 5 tiene
forma cerrada:

$$\int_0^T \|\mu^{(4)}\|^2 dt = \sum_\text{ejes} \left(A^2 T + A B T^2 + \tfrac{1}{3}B^2T^3\right), \quad A = 24 c_4,\ B = 120 c_5.$$

**Elegir con su propia percepción (`--select sgm`, R22 desde 0.22.0).** La cabeza de costo resultó ser el
eslabón débil: con las mismas 3 hipótesis, un **oráculo** que elige por la holgura verdadera de cada
polinomio volado llega mucho más. La regla `sgm` hace lo mismo sin trampa, con lo que el dron mide:
1. cada cuadro de SGM se retroproyecta a puntos del mundo con la pose estimada del dron (la fórmula de
   retroproyección de §5.3);
2. los puntos se agrupan en celdas de 0.15 m; cuenta como obstáculo la celda vista en al menos 2 de los
   ~15 cuadros del último segundo (el moteado del SGM no se repite de un cuadro a otro);
3. la holgura de un plan es la distancia mínima de su polinomio (21 muestras en 1 s) a esas celdas;
4. entre los planes con holgura $\ge r_q + 0.15$ m se toma el de menor snap; si no hay ninguno, el de más
   holgura.

Es un híbrido: la red **propone** y una regla geométrica **dispone**. No usa nada que el dron no tenga.

**No soltar un plan libre (`--keep`, desde 0.24.0).** En cada cuadro el plan en curso se reemplaza por una
de las 3 hipótesis nuevas. Si ninguna está libre pero lo que queda del plan en curso (≥ 0.3 s) sí, `--keep`
sigue volando ese plan; nunca cambia un plan libre por uno que no lo está. Es la regla clásica del
horizonte deslizante: conservar la última trayectoria factible como respaldo. Cada cuadro le da a la red
otra oportunidad desde un punto de vista nuevo.

**Lo que no funcionó: `--lookahead`.** Exigir que también la continuación del plan a su velocidad final
esté libre (0.5–1 s más) no cambió nada (53 y 54/90 frente a 58/90). Los choques no venían de planes que
terminan apuntando a un tronco.

| selección (almacén, 90 vuelos) | 3 m/s | 5 m/s | 7 m/s | total |
|---|---|---|---|---|
| `snap` (el paper) | 8/30 | 13/30 | 17/30 | 38/90 |
| `oracle` (privilegiado, solo diagnóstico) | 16/30 | 21/30 | 22/30 | 59/90 |
| `sgm` | 12–17/30 | 21–22/30 | 24/30 | 58–62/90 |
| **`sgm --keep`** | **21/30** | **24/30** | **24/30** | **69/90** |

### 4.7 Control y dinámica

- **Vehículo:** esfera de 0.2 m, empuje máximo normalizado $c_\text{max} = 35.3$ m/s² (relación
  empuje/peso 3.6) e inercia $J = 0.007$ kg·m², los valores del paper (`agile/platform.py`). El
  simulador integra cuerpo rígido con motores de primer orden.
- **Control en cascada** (`control/controller.py`): velocidad (PI + prealimentación de aceleración) →
  vector de empuje y orientación deseada en SO(3) (Lee et al.) → tasas angulares → torques → mezclador
  con saturación. La referencia del plan entra como
  $v_\text{cmd} = v_\text{ref} + K_p (p_\text{ref} - p)$, con $K_p = 3$, a 500 Hz. La cámara y la red
  funcionan a 15 Hz, con 50 ms de latencia entre captura y plan.
- **Diferencia con el paper:** el paper sigue la trayectoria con un MPC (empuje y tasas a 100 Hz). El
  controlador por velocidad de aquí sigue peor a alta velocidad. Es una de las causas probables de la
  distancia que queda a 10 m/s.

### 4.8 DAgger: por qué no basta con imitar

**El problema del desplazamiento de distribución.** Si el estudiante solo se entrena con los estados que
visita el experto (*behavior cloning*), en cuanto comete un error pequeño llega a estados que nunca vio,
se equivoca más, y el error se acumula. Ross, Gordon y Bagnell (2011) acotan el costo esperado en un
horizonte $T$ con error de imitación $\epsilon$ por paso:

$$J(\hat\pi_\text{BC}) \le J(\pi^*) + T^2 \epsilon \qquad\text{frente a}\qquad J(\hat\pi_\text{DAgger}) \le J(\pi^*) + u\,T\,\epsilon + O(1).$$

Clonar es cuadrático en el horizonte; DAgger es lineal.

**DAgger** (*dataset aggregation*): se vuela con el estudiante, se etiquetan **los estados que visita el
estudiante** con el experto, se agregan al conjunto y se reentrena. Así el estudiante aprende a corregirse
desde sus propios errores.

**La variante del paper:** el estudiante vuela mientras esté a menos de $\xi$ metros de $\tau_\text{gbl}$;
si se aleja más, toma el control $\tau_\text{gbl}$ y lo devuelve. $\xi$ empieza en 0 y crece
($\xi \leftarrow \min(\xi + 0.25, 6)$ cada 30 entornos). Con $\xi$ pequeño, el estudiante solo explora
cerca de lo correcto y los datos son útiles. Con $\xi$ grande, ve estados más alejados y aprende a volver.
Un $\xi$ que queda bajo produce un estudiante que **nunca vio** cómo recuperarse de una deriva grande. Esa
es la explicación más probable de la deriva lateral observada en las dos primeras rondas en Flightmare.

### 4.9 Cómo se mide, y cuánto se puede creer

El éxito es llegar a la meta sin choque. Con $n$ vuelos, el intervalo de confianza del 95 % de Wilson es

$$\frac{\hat p + \frac{z^2}{2n} \pm z\sqrt{\frac{\hat p(1-\hat p)}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}}, \qquad z = 1.96.$$

Con 10 vuelos el intervalo es ancho: 5/10 da $[0.24, 0.76]$, y 7/10 da $[0.40, 0.89]$. Diferencias de 1 o
2 éxitos en 10 no son concluyentes. Por eso las comparaciones importantes se hacen con 40 vuelos (10 por
velocidad) en R18 y con 90 (30 por velocidad) en R22, y aun así 23/40 da $[0.42, 0.71]$.

**Comparar emparejado (McNemar).** Dos variantes vuelan los mismos entornos. Lo que informa son los
entornos **discordantes**: $b$ donde solo llega la variante nueva y $c$ donde solo llega la otra. Bajo la
hipótesis de que no hay diferencia, $b \sim \text{Binomial}(b + c, \tfrac12)$, y el valor $p$ exacto es

$$p = \min\Big(1,\ 2\sum_{i=0}^{\min(b,c)} \binom{b+c}{i}\, 2^{-(b+c)}\Big).$$

Los entornos donde ambas llegan o ambas chocan no dicen nada sobre la diferencia, y quitarlos es lo que
hace la prueba más sensible que comparar dos proporciones (`agile.stats`, `r22 compare`).

**El ruido entre corridas es grande.** Dos reglas casi iguales discrepan en ~30 de 90 entornos: un mismo
entorno llega en una corrida y choca en otra. Con ~30 discordantes, McNemar solo ve diferencias de unas
12 llegadas de 90. Una diferencia menor, por plausible que sea, no está medida.

**Tasa de choque por segundo y por metro.** Con $k$ choques, tiempo total de vuelo $T$ y distancia $D$,
la tasa es $k/T$ o $k/D$ (el estimador de máxima verosimilitud de un proceso de Poisson). Distinguir las dos
dice de qué depende el fallo:
- si va **por metro**, el fallo depende de los obstáculos que se cruzan;
- si va **por segundo**, depende de las decisiones, que son 15 por segundo a cualquier velocidad.

En R22, sin `--keep`, la tasa por segundo era casi la misma a 3, 5 y 7 m/s con cualquier selección (`sgm`
5.4, 5.0 y 3.4 %/s). Por eso 3 m/s era lo peor: el vuelo dura 15 s y no 6. Con `--keep` pasa a 2.3, 2.6 y
3.8 %/s y ya crece con la velocidad, como cabe esperar.

**Resultados de R18 en nuestro simulador** (bosque de prueba, $\delta = 1/25$, semillas 1000–1009):

| piloto | 3 m/s | 5 m/s | 7 m/s | 10 m/s | total (IC 95 %) |
|---|---|---|---|---|---|
| experto M-H (privilegiado) | 10 | 10 | 10 | 10 | 40/40 [0.91, 1.00] |
| estudiante, cabeza promedio (`student_train`) | 3 | 6 | 7 | 6 | 22/40 [0.40, 0.69] |
| estudiante, cabeza espacial (`student_spatial_in`) | 7 | 9 | 6 | 1 | 23/40 [0.42, 0.71] |
| estudiante (`student_train`) eligiendo con `--select sgm` | 6 | 8 | 7 | 7 | 28/40 [0.55, 0.82] |
| ciego (recto a la meta) | | | | | 16/40 [0.26, 0.55] |
| paper, estudiante (bosque simulado) | 10 | 10 | — | 6 | 100 % hasta 5 m/s, 60 % a 10 m/s |

---

## 5. Simuladores

### 5.1 El simulador propio

Un renderizador por trazado de rayos en numpy (`scene/renderer.py`) con primitivas (cilindros, cajas,
esferas, plano), texturas procedurales e iluminación aleatoria. Genera el par estéreo, al que se aplica
la misma exposición y desenfoque a ambas imágenes y ruido independiente a cada una. Los obstáculos son
primitivas exactas, así que el campo de distancias $d(p)$ se calcula analíticamente
(`agile/geometry.py`), por ejemplo para un cilindro:
$d = \sqrt{\max(d_{xy},0)^2 + \max(d_z,0)^2} + \min(\max(d_{xy}, d_z), 0)$. Es lento (~0.25 s por par),
pero es exacto y se puede correr en paralelo en todos los núcleos.

### 5.2 Gazebo (R21)

Gazebo Harmonic + ROS 2 Jazzy en Docker. El mismo mundo de prueba se exporta como SDF; un X3 ágil lleva
el par estéreo (320 × 240, 15 Hz). El nodo `gazebo/nodes/agile_node.py` corre SGM, el estudiante en
TFLite y el seguimiento con avance por retardo ($v_\text{cmd} = v_\text{ref} + \tau a_\text{ref} + k_p (p_\text{ref} - p)$).
Sirve para probar la política con física y middleware reales, no para entrenar.

### 5.3 Flightmare (R22): el simulador del paper

Flightmare es un reproductor de Unity, con escenas fotorrealistas, que recibe poses por ZMQ y devuelve
imágenes. El cliente es nuestro (`flightmare/bridge.py`, sin compilar flightlib) y reproduce el protocolo
del reproductor (código fuente público: `flightmare_unity`, rama `dev/agile_flight`).

**Convenciones de ejes.** ROS es dextrógiro (x adelante, y izquierda, z arriba). Unity es levógiro (x
derecha, y arriba, z adelante). La conversión de flightlib intercambia y↔z:

$$p_U = (x, z, y), \qquad q_U = (-q_x, -q_z, -q_y, q_w), \qquad T_U = M\,T\,M^\top,\ M = \begin{bmatrix}1&0&0&0\\0&0&1&0\\0&1&0&0\\0&0&0&1\end{bmatrix}.$$

Intercambiar dos ejes cambia la orientación del sistema, de dextrógiro a levógiro. Por eso el mundo no
queda reflejado. `r22 check` lo verifica con cajas fuera del eje.

**Profundidad de Unity.** El shader escribe $z/z_\text{far}$ (`Linear01Depth`), así que en metros es
valor × plano lejano (100 m). El probe lo confirmó contra el SGM: SGM / valor crudo = 96, frente a 100
esperado.

**La focal tiene que ser la misma (0.21.1).** El reproductor de Unity recibe el campo de visión
**vertical**. Para la focal del rig ($f = 320$ px a 640 × 480, HFOV 90°) es

$$\text{VFOV} = 2\arctan\frac{H/2}{f} = 2\arctan\frac{240}{320} = 73.74°.$$

Con 90° vertical, la focal de Unity era 0.75 × la nuestra. Las imágenes eran coherentes entre sí y el SGM
funcionaba, pero cada píxel quedaba en otro ángulo del que suponía la red, el retroproyector y el juez.
Corregirlo llevó al estudiante de 9 a 20 de 30 vuelos. `r22 check` mide ahora la focal ajustando las caras
de cajas conocidas en la profundidad de Unity (cociente 0.99).

**Cámara inclinada a alta velocidad.** Para acelerar o vencer el arrastre, el dron se inclina hacia
adelante un ángulo $\theta \approx \arctan(a_x/g)$. A 7–10 m/s, $\theta$ ronda 20–30°. Con un campo de
visión vertical de 74° (±37°), una cámara fija mira al suelo y solo ve
$37° - 30° \approx 7°$ por encima del horizonte. El paper inclina el sensor 30° hacia arriba a partir de
7 m/s. Aquí se aplica por episodio desde 0.20.1 (`set_tilt`). Antes no se aplicaba en ningún simulador,
lo que explica en parte el 1/10 a 10 m/s de R18.

**El mapa privilegiado en Flightmare.**
1. Unity puede exportar una nube de puntos, pero solo de los objetos con collider. El bosque de 2022 casi
   no tiene: la nube salió como una lámina plana bajo el río.
2. Se probó entonces reconstruir el mapa con la propia profundidad de Unity desde una malla de poses
   (`r22 scan`). Cada píxel con profundidad $Z$ se retroproyecta:
   $$P = O + R_c\,\left[1,\ -\tfrac{u + 0.5 - c_x}{f},\ -\tfrac{v + 0.5 - c_y}{f}\right]^\top Z.$$
   Los puntos se agrupan en vóxeles. Como la superficie real está a lo sumo a media diagonal del centro
   del vóxel, la distancia al centro ocupado más cercano, menos $r/2$, es una cota conservadora del
   despeje. En la vegetación natural falla: las hojas salen como triángulos sólidos, faltan zonas no
   vistas (13 % de error) y el agua es transparente.
3. La solución final copia la del paper: una **arena**. Se escanea una vez la escena vacía (el almacén
   de agile_flight: piso, paredes, techo, sin vegetación). Su ocupación se convierte en un campo de
   distancias con la transformada de distancia euclidiana exacta (`scipy.ndimage.distance_transform_edt`,
   `arena.GridSDF`). En cada entorno se **siembran troncos** de geometría conocida: la caja de madera
   `rpg_box02`, escalada a 0.3–0.8 × 0.3–0.8 × 6 m, que Unity mueve en cada cuadro como objeto dinámico.
   El mapa es la unión exacta de ambos:
   $$d(p) = \min\big(d_\text{escena}(p),\ \min_j d_{\text{caja}_j}(p),\ p_z - z_\text{suelo}\big).$$

**Por qué el almacén y no el bosque.** El experto y el juez necesitan geometría exacta, y el bosque de
2022 no la tiene. La red ve profundidad, no texturas, y la geometría de prueba (troncos verticales de
0.5–0.7 m, $\delta = 1/25$) es la del paper. El bosque queda para probar después si lo aprendido
generaliza.

**DAgger en Flightmare** (`flightmare/dagger.py`). Hay un solo reproductor de Unity, así que los vuelos
van en serie. El experto, la parte lenta (~0.4 s por etiqueta), se calcula en un grupo de procesos
mientras vuela el siguiente entorno (`Rollout.defer`). El cliente mantiene vivo al reproductor con un
cuadro cada 2 s, porque vuelve a su menú tras 10 s sin mensajes, por ejemplo mientras la red entrena
entre rondas.

---

## 6. Infraestructura del proyecto

### 6.1 Estructura del repositorio

```
nanouav/
├── src/nanouav/              el paquete (código en inglés)
│   ├── config.py paths.py settings.py   convenciones de profundidad, rutas, banderas NANOUAV_*
│   ├── scene/                renderizador propio (World, primitivas, texturas, render_view)
│   ├── sensors/              ToF 8×8, flujo óptico
│   ├── perception/           CNN int8, zoo de modelos, calibración y fusión con ToF
│   ├── planning/             EvasionPlanner, MHPlanner, políticas, misiones
│   ├── control/              vehículo, simulador de cuadricóptero, controlador en cascada, supervisor
│   ├── sim/                  episodios en lazo cerrado, misiones, escenarios (compartidos con Gazebo)
│   ├── training/             entrenamiento de las CNN de profundidad
│   ├── agile/                R18: stereo (SGM), envs, expert (τ_gbl + M-H), trajectory, policy,
│   │                         rollout, dagger, diagnose, video, gz_core, platform
│   ├── follow/               R19–R20: detectores, tracker, misiones de seguimiento
│   ├── flightmare/           R22: bridge (cliente ZMQ), fake (reproductor falso), cloud, scan,
│   │                         arena, dagger, run (CLI r22)
│   ├── benchmark/            latencia, energía, jitter del hardware objetivo
│   ├── experiments/          un módulo por experimento + registro (__init__.py)
│   ├── results/              informes (tasas con IC de Wilson)
│   └── cli.py runner.py      CLI `nanouav` y suites en paralelo
├── gazebo/                   mundos, modelos, nodos ROS 2, scripts (gz_docker.sh, run_agile.sh)
├── docker/                   imágenes base y de simulación (Harmonic + Jazzy)
├── configs/suites/           lotes de experimentos en YAML
├── tests/                    pytest (unos 175 tests: planificadores, R18, Flightmare con reproductor falso, …)
├── docs/                     arquitectura, experimentos, Gazebo, GPU, este documento, research/R*.md
├── models/ data/ results/    fuera de git: modelos, datasets, resultados
├── CHANGELOG.md              cada versión, y si cambia algún resultado
└── pyproject.toml            paquete instalable; extras: tflite, train, dev, flightmare
```

Las dependencias van en un solo sentido:
`config/paths/settings ← scene, sensors ← perception ← planning ← control ← sim ← experiments`.
`agile`, `follow` y `flightmare` se apoyan en `scene` y `control`, pero no en el pipeline del nano.

### 6.2 La CLI y el registro de experimentos

Cada experimento es un módulo con su propia línea de comandos, registrado con un identificador estable
(`experiments/__init__.py`):

```bash
nanouav list                                   # experimentos disponibles
nanouav run r18 eval --model results/r18/student_train.keras
nanouav run r22 dagger --scene warehouse --init results/r18/student_spatial_in.keras
nanouav suite configs/suites/baseline.yaml -j 7 --resume
nanouav summarize --dir results/baseline       # tasas de éxito con IC de Wilson
nanouav doctor                                 # dependencias, rutas y modelos
```

Los subcomandos de R18 y R22:

| `nanouav run r18 …` | qué hace |
|---|---|
| `sgm` | calidad del SGM frente a la profundidad verdadera |
| `pilots` | pilotos de referencia: experto, τ_gbl, ciego |
| `dagger` | recolección y entrenamiento (850 entornos, rondas de 30) |
| `train` | reentrenar con los datos acumulados (`--head spatial`, `--best`, `--val-by-env`) |
| `eval` | bosque de prueba a 3, 5, 7 y 10 m/s |
| `diagnose` | clasificación de cada choque |
| `video` | video de un vuelo |
| `export` | TFLite para Gazebo o hardware |

| `nanouav run r22 …` | qué hace |
|---|---|
| `probe` | convenciones de Unity en un cuadro |
| `arena` | escaneo de la escena vacía, suelo, zona libre y caja de prueba |
| `check` | geometría con cajas fuera del eje: espejo, inclinación, SGM |
| `dagger` | DAgger con imágenes de Unity, reanudable |
| `eval --arena` | evaluación en los bosques de prueba (`--select`, `--keep`; tasas por segundo y por metro) |
| `compare A.json B.json` | dos evaluaciones emparejadas por entorno (McNemar) |
| `diagnose` | clasificación de choques, deriva, calidad del SGM y errores de su propia percepción |
| `video --arena` | video de un vuelo: cámara con túnel, SGM, mapa local con su propio SGM, costos |
| `scan`, `survey`, `cloud`, `lines` | mapas de escenas sin arena |
| `fake` | reproductor falso para probar sin Unity |

### 6.3 Flujo de datos de un vuelo con el estudiante

```
Unity (Flightmare) ── par estéreo VGA ──►  SGM (OpenCV) ──► profundidad 480×640 ──► 120×160
        ▲                                                                              │
        │ pose (ZMQ, 15 Hz)                         estado (v, R, dirección a la meta) │
        │                                                                              ▼
 simulador de cuadricóptero ◄── control en cascada (500 Hz) ◄── plan polinómico ◄── estudiante (TF)
   (dinámica del paper)          v_cmd = v_ref + K_p·Δp           Eq. 8 + β + elección   3 hipótesis + costos
                                                                     ▲
                       elección: holgura a las celdas de su propio SGM (último segundo) · --keep

 solo para entrenar y juzgar:  mapa exacto (arena) ──► experto M-H (etiquetas) · juez de choques
```

### 6.4 El cliente de Flightmare

```
nanouav (Python)                                   RPG_Flightmare.x86_64 (Unity)
  PUB  tcp://*:10253  ── ["Pose", json] ────────►  configuración (escena, cámaras, objetos) y una pose por cuadro
  SUB  tcp://*:10254  ◄── [json, imágenes…] ─────  {"ready": true}, luego RGB izq/der + profundidad
  PUB  ["PointCloud", json] ───────────────────►  escribe un .ply (solo colliders)
```

Nuestro lado abre los sockets y Unity se conecta. Con `--unity` o `NANOUAV_FLIGHTMARE`, cada comando abre
su propio reproductor y lo cierra al terminar. El reproductor falso (`flightmare/fake.py`) habla el
mismo protocolo con nuestro renderizador e imita lo que hace el real: profundidad dividida por el plano
lejano, nube absoluta, objetos dinámicos y reinicio tras 10 s sin mensajes. Con él corren los tests de
Flightmare sin Unity.

### 6.5 Cómo se desarrolla

- **Parches:** cada cambio es un commit exportado como `nanouav_00NN_*.patch` y se aplica con `git am`
  en la máquina de trabajo (Python 3.10, TITAN X Pascal, 8 núcleos). Si un `git am` falla, `git am --quit`
  limpia el estado sin deshacer lo ya aplicado.
- **Versiones:** cada parche sube la versión (SemVer) y el `CHANGELOG.md` dice si cambia algún resultado
  numérico.
- **Tests:** cada pieza nueva llega con tests que corren sin GPU ni Unity (reproductor falso, modelos
  diminutos). Antes de cada parche se corre la batería completa.
- **Informes:** cada iteración tiene su `docs/research/RNN_*.md`, con la pregunta, qué se toma de la
  fuente, el comando exacto, la tabla con IC y la conclusión.
- **Lo que no va en git:** modelos (`models/`, `results/**.keras`), datasets (`data/`) y resultados
  (`results/`).

---

## 7. Estado actual y lo que sigue

**Dónde estamos (R22, 0.24.0).** El estudiante de 300 entornos llega en **69 de 90 vuelos** en el almacén:
21/30, 24/30 y 24/30 a 3, 5 y 7 m/s. Elige con su propio SGM (`sgm`) y no suelta un plan libre (`--keep`).
El experto privilegiado y τ_gbl llegan siempre.

**Lo que se aprendió, en orden:**
1. **Geometría antes que aprendizaje.** El mal estudiante inicial era sobre todo una focal equivocada en
   Unity (§5.3): de 9 a 20 de 30 vuelos al corregirla.
2. **Las etiquetas deben ser función de lo que ve el estudiante** (R23). Si las etiquetas siguen una ruta
   que no ve ($\tau_\text{gbl}$) mientras su dirección $\omega$ apunta a la recta, se fija un piso a la
   pérdida. Las etiquetas replaneadas (`ctg`) aprenden mejor, pero en Flightmare todavía no vuelan mejor:
   sin confirmar.
3. **La cabeza de costo era el cuello de botella**, no las hipótesis. Una regla geométrica sobre su propio
   SGM elige tan bien como el oráculo (62 contra 59/90) y mucho mejor que la regla del paper (38/90,
   McNemar $p = 4 \cdot 10^{-5}$).
4. **Más DAgger con la misma receta no mejora:** 450 entornos dan lo mismo que 300.
5. **Los choques iban por segundo, no por metro.** El fallo estaba en las decisiones: soltar un plan libre
   cuando ninguna hipótesis nueva lo está. `--keep` lo corrige: +9 a 3 m/s, donde hay más decisiones por
   metro.

**Lo que queda:**
- **7 m/s.** 5 de sus 6 choques llegan en los primeros 2.5 s: el vuelo arranca ya a la velocidad objetivo,
  con troncos desde 2.5 m y sin historia de percepción. Hay que medir si es un artefacto del protocolo de
  salida (`r22 diagnose` con `--keep`).
- **Confirmar `--keep`** con una segunda corrida o con más entornos: contra la corrida `sgm` del mismo
  modelo, $p = 0.07$.
- **10 m/s, el bosque de Unity sin reentrenar, y destilar al nano-UAV:** una red del tamaño de PULP-Dronet
  (imagen en grises + ToF 8 × 8).

**Diferencias que quedan con el paper**, todas documentadas:
- seguimiento con controlador de velocidad, no MPC;
- obstáculos con forma de caja en un almacén, no árboles en un bosque;
- elección por la geometría de su propio SGM y no por la cabeza de costo;
- 300 entornos de entrenamiento en Flightmare, frente a 850 en nuestro simulador.

---

## 8. Referencias

- A. Loquercio, E. Kaufmann, R. Ranftl, M. Müller, V. Koltun, D. Scaramuzza. *Learning high-speed flight in
  the wild.* Science Robotics 6(59), 2021.
- D. Chen, B. Zhou, V. Koltun, P. Krähenbühl. *Learning by cheating.* CoRL 2019.
- S. Ross, G. Gordon, D. Bagnell. *A reduction of imitation learning and structured prediction to no-regret
  online learning (DAgger).* AISTATS 2011.
- H. Hirschmüller. *Stereo processing by semiglobal matching and mutual information.* IEEE TPAMI 30(2), 2008.
- D. Mellinger, V. Kumar. *Minimum snap trajectory generation and control for quadrotors.* ICRA 2011.
- T. Lee, M. Leok, N. H. McClamroch. *Geometric tracking control of a quadrotor UAV on SE(3).* CDC 2010.
- E. B. Wilson. *Probable inference, the law of succession, and statistical inference.* JASA 22, 1927.
- Y. Song, S. Naji, E. Kaufmann, A. Loquercio, D. Scaramuzza. *Flightmare: a flexible quadrotor simulator.*
  CoRL 2020. Código del reproductor: github.com/uzh-rpg/flightmare_unity (rama `dev/agile_flight`).
- D. Falanga, S. Kim, D. Scaramuzza. *How fast is too fast? The role of perception latency in high-speed
  sense and avoid.* IEEE RA-L 4(2), 2019.
- L. Lamberti et al. *Distilling tiny and ultra-fast deep neural networks for autonomous navigation on
  nano-UAVs* (PULP-Dronet v3). IEEE IoT Journal, 2024.
- Q. McNemar. *Note on the sampling error of the difference between correlated proportions or percentages.*
  Psychometrika 12(2), 1947.
- Informes del proyecto: `docs/research/README.md` (índice), `R18_agile_autonomy.md`,
  `R21_gazebo_agile.md`, `R22_flightmare.md`, `R23_direccion_y_etiquetas.md`; arquitectura del pipeline
  del nano: `docs/arquitectura.md`.
