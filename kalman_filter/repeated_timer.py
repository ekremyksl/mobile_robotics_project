from threading import Timer


class RepeatedTimer(object):
    def __init__(self, periode, function, *args, **kwargs):
        self._timer     = None
        self.periode   = periode
        self.function   = function
        self.args       = args
        self.kwargs     = kwargs

    def start(self):        
        self._timer = Timer(self.periode, self._timer_callback)
        self._timer.start()
        self.is_running = True

    def stop(self):
        self._timer.cancel()
        
    def _timer_callback(self): # call _timer_callback only from self.start to not restart timer before it reached its end
        self.start() # recall start and recreate new timer for next periode
        self.function(*self.args, **self.kwargs)