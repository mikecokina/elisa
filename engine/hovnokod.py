class A(object):
    def __init__(self, a):
        self._a = a
        self.ref = None

    @property
    def a(self):
        return self._a

    @a.setter
    def a(self, var):
        self._a = var
        self.callback()

    def callback(self):
        if self.ref is not None:
            self.ref.__init__(self)


class B(object):
    def __init__(self, obj):
        self.obj = obj
        self.obj.ref = self

        print("Init")

x = A(a=3)
x.a = 11
y = B(obj=x)
x.a = 10


