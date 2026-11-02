"""Isolated real application with timing/array instrumentation (no fake fitters)."""
import functools
import inspect
import json
import os
import signal
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
from elisa import settings
from elisa.analytics.tasks import AnalyticsTask
from elisa.analytics.params.parameters import BinaryInitialParameters
from elisa.analytics.binary_fit.plot import LCPlot, RVPlot
from elisa.ui.app import build_app
from elisa.ui.tabs.lc_fitting.components.param_inputs import FIELD_ORDER as LC_FIELDS
from elisa.ui.tabs.rv_fitting.components.param_inputs import FIELD_ORDER as RV_FIELDS

settings.configure(NUMBER_OF_PROCESSES=1, NUMBER_OF_MCMC_PROCESSES=1)

OUT = Path(os.environ['E2E_WORK'])
original_fit = AnalyticsTask.fit


def write_event(event):
    with (OUT / 'backend.jsonl').open('a') as stream:
        stream.write(json.dumps(event, default=str) + '\n')


@functools.wraps(original_fit)
def measured_fit(self, x0, **kwargs):
    lc = self.__class__.__name__.startswith('LC')
    if self.method == 'least_squares':
        kwargs.setdefault('max_nfev', 3)
    if lc:
        atmosphere = {'primary': 'bb', 'secondary': 'ck04'}
        self.fit_cls.atmosphere_model = atmosphere
        self.fit_cls.fit_method_instance.atmosphere_model = atmosphere
        kwargs.setdefault('discretization', 10)
    if self.method == 'mcmc':
        assert 0 < kwargs['nsteps'] <= 10
        assert kwargs['burn_in'] == 0
    np.random.seed(42)
    start = time.perf_counter()
    event = {'kind': 'LC' if lc else 'RV', 'method': self.method, 'options': kwargs}
    write_event({'started': event})
    try:
        initial = BinaryInitialParameters(**x0) if isinstance(x0, dict) else x0
        event['input_keys'] = list(initial.data)
        event['fixed'] = {key: param.value for key,param in initial.get_fixed().items()}
        event['free'] = list(initial.get_fitable())
        event['constraints'] = {key: param.constraint for key,param in initial.get_constrained().items()}
        result = original_fit(self, x0, **kwargs)
        event.update(seconds=time.perf_counter() - start, result=result,
                     data_shapes={k: list(v.y_data.shape) for k, v in self.data.items()})
        if self.method == 'mcmc':
            chain = np.asarray(self.fit_cls.flat_chain)
            event.update(chain_shape=list(chain.shape), chain_finite=bool(np.isfinite(chain).all()))
        write_event(event)
        return result
    except Exception as exc:
        write_event({**event, 'seconds': time.perf_counter()-start, 'error': repr(exc)})
        raise


AnalyticsTask.fit = measured_fit

# These supported plotting options avoid a 300-point, 5-degree display model.
# The original model calculation and plotting implementation still execute.
def instrument_plot(cls):
    original = cls.model
    @functools.wraps(original)
    def model(self, *args, **kwargs):
        kwargs.setdefault('number_of_points', 40)
        if cls is LCPlot:
            kwargs.setdefault('discretization', 10)
        figure = original(self, *args, **kwargs)
        if figure is not None:
            arrays = [np.asarray(line.get_ydata(), dtype=float) for ax in figure.axes
                      for line in ax.lines if len(line.get_ydata()) == 40]
            write_event({'plot': cls.__name__, 'shapes': [list(a.shape) for a in arrays],
                         'finite': bool(arrays) and all(np.isfinite(a).all() for a in arrays)})
        return figure
    cls.model = model


instrument_plot(LCPlot)
instrument_plot(RVPlot)
app = build_app()
manifest = {'components': {}, 'callbacks': {}}
for ident, component in app.blocks.items():
    if hasattr(component, 'elem_id') and not component.elem_id:
        component.elem_id = f'e2e-{ident}'
    manifest['components'][str(ident)] = {
        'selector': '#' + str(getattr(component, 'elem_id', '')),
        'type': component.__class__.__name__, 'label': getattr(component, 'label', None),
        'value': getattr(component, 'value', None),
    }
for fn in app.fns.values():
    if fn.fn is None:
        continue
    closure = inspect.getclosurevars(fn.fn).nonlocals
    name = fn.fn.__name__
    if name == '_unified_lsqrt':
        manifest['callbacks']['LC'] = {
            'fields': dict(zip(LC_FIELDS, [c._id for c in fn.inputs[-len(LC_FIELDS)-1:-1]], strict=True)),
            'inputs': [c._id for c in fn.inputs],
        }
    if name == 'handler' and 'fit_keys' in closure and 'data_keys' in closure and len(fn.inputs) == 3 + len(RV_FIELDS):
        manifest['callbacks']['RV'] = {
            'fields': dict(zip(RV_FIELDS, [c._id for c in fn.inputs[3:]], strict=True)),
            'inputs': [c._id for c in fn.inputs],
        }
(OUT / 'manifest.json').write_text(json.dumps(manifest, default=str))
def shutdown(_signum, _frame):
    app.close()
    raise SystemExit(0)


signal.signal(signal.SIGTERM, shutdown)
app.launch(server_name='127.0.0.1', server_port=int(sys.argv[1]),
           inbrowser=False, share=False, show_error=True, allowed_paths=[str(OUT)])
