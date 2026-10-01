# Shang et al. · *Fully Autonomous Neuromorphic Navigation and Dynamic Obstacle Avoidance* (NeurIPS 2025, slides)

Cámara de eventos + chip Speck (SNN) + Jetson Orin NX. Aportes:
1. **Visual homing con auto-calibración.** Snapshots de eventos en puntos de paso; CalibNet
   compara el tensor actual con la snapshot y corrige la deriva inercial (δr ∝ t², t³) dentro de
   un área de captura de ~20 cm.
2. **Modelo de campo receptivo de anuros:** F = min(A₁K(t,τₑ)G, E_th) − min(A₂K(t−Δt,τᵢ)G, I_th).
   La excitación menos la inhibición retardada suprime lo estático y deja solo lo dinámico; un
   campo de potencial difuso da la dirección de escape. Latencia 2.34 ms, éxito 94.5 %.
3. **Energía:** Speck consume 15 mW; el cómputo baja al 5 % y el sistema de cómputo al 21 %.

Notas críticas: el "tiny UAV" pesa 856 g y el Speck necesita un x86 de soporte. Las diapositivas
no dan el protocolo de los obstáculos (velocidades, ángulos) ni el número de ensayos. La
reducción de energía es del cómputo, no del vuelo: con los números de Su et al. 2026, el cómputo
ya es ~1 % de la potencia de vuelo de un nano.

## R9 · Deriva: resultado negativo, pero corrige un diagnóstico propio
- **Alineación observación→mapa local** (`DRIFT_CORR`): empeora la pose (RMS x 1.6 → 2.1 m).
- **Keyframe de hace 1 s + estimación de sesgo** (`VO`): tampoco mejora, ni integrando el sesgo
  ni corrigiendo la posición a 1 Hz.
- **La traza con la pose verdadera corrige R8:** la deriva real llega a 1.6–2.5 m. El fallo no
  era el mapa "embarrado": el controlador sigue una velocidad sesgada y el vehículo se desplaza
  físicamente. Con τ = 1 s, el "sesgo" es ruido coloreado; ninguna alineación de nubes CNN+ToF
  8×8, ruidosas y escasas, lo compensa.
- **Requisito que se mantiene:** error de velocidad ≤ 0.05 m/s (10/10 en R8). La lección del
  paper aplica a misiones con revisita (patrulla, regreso a casa): relocalizar contra snapshots
  de lugares conocidos, no hacer odometría con el mismo sensor.
- Flags apagados por defecto; regresión verificada bit a bit.

## R10 · Obstáculos dinámicos (`r10_dynamic.py`, nano, v = 1 m/s, balón r = 0.12 m, n = 5 por celda)

| variante | de frente (0.5/1/2/3 m/s) | cruzando |
|---|---|---|
| base (consistent) | **4/20** (4/0/0/0), 13 golpes | 17/20 |
| + capa ERF/IRF (`RF_DYN`) | 8/20 (5/3/0/0) | 17/20 |
| + perfil agresivo solo en EVADE (`EVADE_AGGR`) | **14/20** (4/5/5/0) | 17/20 |

Capa `RF_DYN`, adaptada de eventos a profundidad:
- **Excitación:** celda ocupada ahora que antes se vio libre.
- **Inhibición:** lo ya mapeado o nunca visto no dispara; así se suprimen lo estático y lo que
  entra al FOV al avanzar.
- **Seguimiento:** grupos → pistas α-β (el balón se detecta a ~2 m y su velocidad converge en
  0.4 s); el volumen predicho de 1.5 s entra al plan, sin estela en la memoria.
- **Escape:** perpendicular a la trayectoria del objeto cuando el punto de máxima aproximación
  predicho < holgura (el análogo del gradiente del campo de potencial).

Hallazgos:
1. Sin predicción, el planificador esquiva donde el objeto *estaba*. Con la predicción en el plan
   pero sin escape, frena y retrocede frente a lo que viene: lo peor posible.
2. **La latencia sí importa para lo dinámico** (en R4, con obstáculos estáticos, importaba poco).
   Confirmación (0.3 s) + latencia (0.16 s) + maniobra a 2.2 m/s² (0.64 s) ≈ 1.1 s de cierre.
   Liberar la aceleración solo durante el escape (6 m/s², inclinación de 40°) sube el éxito de
   frente de 8/20 a 14/20. Cierre de 4 m/s (v_obs = 3) sigue en 0/5: hace falta detectar a
   ≥ 3.2 m, en el límite del ToF.
3. **Geometría de rumbo constante:** un objeto en curso de colisión aparece a un ángulo fijo,
   atan(v_obj/v_dron). Con HFOV = 60°, **todo cruce con v_obj > 0.58·v_dron es invisible durante
   toda la aproximación**. Verificado: a 1 m/s el balón se mantiene a −45° desde 4.2 m hasta
   1.0 m. Los 3 fallos de cruce son idénticos con y sin capa (t = 7.47 s). A 2 m/s (−63°) tampoco
   se ve: el éxito fue suerte (pasó a 0.4 m). Corolario: cuanto más rápido vuela el dron, más
   amenazas laterales entran en su campo visual. Mitigación de hardware: ToF laterales (VL53L5CX
   a ±45° → cobertura ±67°) o una óptica más ancha.

## Pendiente
- n ≥ 30 por celda; perfil X3; objetos que cruzan dentro de un campo de obstáculos estáticos.
- Simular ToF laterales para cerrar el punto ciego de rumbo constante.
- Detección dinámica con 2 frames en lugar de 3, usando también la tasa de cambio de rango del ToF.
