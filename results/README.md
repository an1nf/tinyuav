# results/

Salidas de experimentos (`nanouav run`) y suites (`nanouav suite`). **No se versionan.**

```
results/
  <archivo>.json|png          experimentos sueltos
  <suite>/jobs.json           estado de cada trabajo de una suite
  <suite>/logs/<id>.log       salida de cada trabajo
  gz_runs/*.json              episodios de Gazebo (monitor del perception_node)
  gz_logs/*.log               logs de gazebo/batch_eval.sh
```

Resumen: `nanouav summarize --dir results/<suite>` · `nanouav summarize --gazebo`.
Las cifras que se citan en un informe se copian a `docs/research/` con el comando y el commit.
