

class RunnerInfo:
    ''' A dynamic dict saving temp information during running
    '''
    def __init__(self):
        self._attributes = {}

    def __setattr__(self, name, value):
        if name == '_attributes':
            super().__setattr__(name, value)
        else:
            self._attributes[name] = value

    def __getattr__(self, name):
        if name in self._attributes:
            return self._attributes[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
    
    def __repr__(self):
        attrs = ''.join(f"\n{key}={value!r}" for key, value in self._attributes.items())
        return f"{type(self).__name__}({attrs})"