from matplotlib.animation import FuncAnimation
from tween import tween
import pytweening

class Animation:
    def __init__(self, fps, fig, lines=None, ax=None, timing_fn=None):
        self._fps = fps
        self._fig = fig
        self._fns = []
        self._lines = lines
        self._ax = ax
        self._timing_fn = timing_fn or pytweening.easeInQuad

    def __call__(self, fn):
        self._fns.append(fn)
        return fn
    
    def pause(self):
        self._fns.append(None)

    def show_points(self, x, y, lines=None):
        if lines is None:
            lines = self._lines
        def fn(i):
            n = x.shape[0]
            i = round(i * (n / self._fps))
            lines.set_data(x[:i], y[:i])
        self._fns.append(fn)
        
    def transform(self, x1, x2, y1, y2, lines=None):
        if lines is None:
            lines = self._lines
        xx = tween(x1, x2, self._timing_fn, fps)
        yy = tween(y1, y2, self._timing_fn, fps)
        def fn(i):
            lines.set_data(xx[i], yy[i])
        self._fns.append(fn)
    
    def animate(self):
        if len(self._fns) == 0:
            raise Exception("No functions to animate!")
        def _animate(i):
            fn = self._fns[i // self._fps]
            if fn is not None:
                fn(i % self._fps)

        return FuncAnimation(
            self._fig,
            _animate,
            interval=1000 / self._fps,
            frames=len(self._fns) * self._fps,
            repeat=True,
        )