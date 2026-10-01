"""
R1 · Deployment footprint of LNCA vs. CNN M (untrained: architecture only).

Measures per variant: parameters, MACs, int8 .tflite size, activation arena
(tensor-lifetime bound, same function as train_cnn), single-thread CPU p50 latency without
XNNPACK ('opt') and with reference kernels ('ref', proportional to MACs, closer to an MCU), and the
TFLite ops used (TFLM compatibility). Weights are random: ACCURACY is not
evaluated here (requires training, see LOCAL_GPU.md §8).

Usage: nanouav run r01   → results/r1_lnca_footprint.json
"""
import os, sys, json, time
from nanouav import paths
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import numpy as np

ROOT = str(paths.repo_root())
from nanouav.perception.model_zoo import build_variant, make_interpreter, tflite_arena_kb
from nanouav.perception.lnca_zoo import build_lnca, count_macs

OUT = os.path.join(str(paths.results_dir()), 'r1_tflite')
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(0)


def export(model, path, n=64):
    import tensorflow as tf
    shapes = {i.name.split(':')[0]: tuple(i.shape[1:]) for i in model.inputs}

    def rep():
        for _ in range(n):
            yield {k: rng.uniform(0, 1, (1,) + s).astype(np.float32) for k, s in shapes.items()}

    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = rep
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    data = conv.convert()
    open(path, 'wb').write(data)
    return len(data)


def latency_ms(path, n=200, ref=False):
    import tensorflow as tf
    R = tf.lite.experimental.OpResolverType
    it = tf.lite.Interpreter(model_path=path, num_threads=1,
                             experimental_op_resolver_type=R.BUILTIN_REF if ref else R.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
    it.allocate_tensors()
    for d in it.get_input_details():
        it.set_tensor(d['index'], np.zeros(d['shape'], np.int8))
    for _ in range(30):
        it.invoke()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); it.invoke(); ts.append(time.perf_counter() - t0)
    return float(np.median(ts) * 1e3)


def ops(it):
    return sorted({o['op_name'] for o in it._get_ops_details()})


def run():
    cands = {
        'CNN-M (actual)': lambda: build_variant('M', batch_size=1),
        'CNN-XS (actual)': lambda: build_variant('XS', batch_size=1),
        'LNCA-32 T4': lambda: build_lnca(32, 4, batch_size=1),
        'LNCA-32 T2': lambda: build_lnca(32, 2, batch_size=1),
        'LNCA-24 T4': lambda: build_lnca(24, 4, batch_size=1),
        'LNCA-32ds T4': lambda: build_lnca(32, 4, enc='ds', batch_size=1),
        'LNCA-32 T4 +ToF': lambda: build_lnca(32, 4, tof=True, batch_size=1),
        'LNCA-32 T4 +ToF +det': lambda: build_lnca(32, 4, tof=True, aux_det=True, batch_size=1),
        'LNCA-32 T1 recurrente +ToF': lambda: build_lnca(32, 1, tof=True, state_in=True, batch_size=1),
    }
    res = {}
    for name, fn in cands.items():
        m = fn()
        fn_ = os.path.join(OUT, name.replace(' ', '_').replace('+', 'p').replace('(', '').replace(')', '') + '.tflite')
        kb = export(m, fn_) / 1024
        it = make_interpreter(fn_)
        r = {'params': int(m.count_params()), 'MMAC': count_macs(m) / 1e6, 'tflite_kb': kb,
             'arena_kb': tflite_arena_kb(it), 'cpu_ms_opt': latency_ms(fn_),
             'cpu_ms_ref': latency_ms(fn_, ref=True), 'ops': ops(it)}
        res[name] = r
        print(f"{name:28s} params {r['params']:6d}  MMAC {r['MMAC']:6.2f}  tflite {kb:6.1f} KB  "
              f"arena {r['arena_kb']:6.1f} KB  cpu opt {r['cpu_ms_opt']:.2f} ms  ref {r['cpu_ms_ref']:.2f} ms", flush=True)
    json.dump(res, open(os.path.join(str(paths.results_dir()), 'r1_lnca_footprint.json'), 'w'), indent=1)
    return res


if __name__ == '__main__':
    run()
