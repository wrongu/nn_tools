class Add:
    def __call__(self, x, y):
        return x + y


class Identity:
    def __call__(self, x):
        return x
