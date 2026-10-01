> **Documento histórico** de la estructura plana anterior (`uav_v2`/`uav_v3`). Los comandos y rutas
> ya no son válidos; ver [docs/migracion.md](../../migracion.md) para su equivalente con `nanouav`.

# _06_control — Control en cascada hasta motores

| Archivo | Qué hace |
|---|---|
| `vehicle.py` | Parámetros físicos y matriz de asignación. Perfiles `x3` (copiado del SDF de Gazebo) y `nano` (ESP32-S3, 65 g). |
| `quad_sim.py` | Dinámica 6-DOF: motores 1er orden asimétricos (τ subida/bajada), empuje kf·ω², reacción km·f, arrastre, perturbaciones, suelo. |
| `reference.py` | `JerkLimited` (velocidad con aceleración y jerk acotados, sin sobrepaso) y `YawShaper`. |
| `controller.py` | Velocidad (PI + feed-forward) → actitud SO(3) → tasa (PI escalado por inercia) → mezclador con prioridades → ω motor. |
| `supervisor.py` | Monitor de estabilidad: dispara RECOVER por inclinación, tasa, caída o saturación de roll/pitch; sale con histéresis. |
| `flight_stack.py` | Integra todo; transferencia sin salto (bumpless) al entrar/salir de RECOVER. |

## Diseño
- **Ganancias** derivadas de la constante de tiempo del motor τm (planta dominante):
  `K_rate ≈ 0.45/τm` (margen de fase ≈ 65°), `K_att ≈ K_rate/4`, `K_vel ≈ K_att/3`.
  El mismo código sirve para el X3 (τm ≈ 19 ms) y el nano (≈ 30 ms).
- **Mezclador**: 1) el yaw cede primero, 2) desplazamiento de colectivo (airmode),
  3) escalado de roll/pitch. Los integradores se congelan con saturación o límite de inclinación.
- **Suavidad**: la referencia limita aceleración (1.2 m/s² horizontal) y jerk (3 m/s³);
  la aceleración de yaw se limita a lo que el vehículo puede sostener
  (`yaw_accel_limit`: X3 ≈ 0.57 rad/s² — km pequeño, T/W 1.47).
- **Recuperación**: v_ref ← v medida (sin salto), velocidad horizontal → 0, altura
  sostenida, yaw congelado, inclinación máx 15°, integrador de velocidad congelado.
  Sale tras 0.5 s con inclinación < 5°, |ω| < 0.3 rad/s, |v_xy| < 0.15 m/s.

## En el sistema real
ESP32-S3: `FlightStack.update` en el núcleo 0 a 500 Hz; percepción + `EvasionPlanner`
en el núcleo 1 a ~10 Hz, que solo envía `set_target(v, yaw)`.
Gazebo: `gazebo/scripts/flight_controller_node.py` (modo `motors`) corre este mismo stack.
