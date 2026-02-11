import functools
import traceback


def print_exceptions(f):
    """Print any exception raised by the annotated function to stderr

    This is helpful if an outer function swallows exceptions, such as the hydra's
    submitit launcher.
    """
    return functools.wraps(f)(ExceptionPrinter(f))


class ExceptionPrinter:
    def __init__(self, f):
        self.f = f

    def __call__(self, *args, **kwargs):
        try:
            return self.f(*args, **kwargs)
        except Exception as e:
            traceback.print_exception(e)
            raise

    def __getattr__(self, attr):
        # This method is called during unpickling, e.g. when submitting a job to slurm,
        # before restoring its internal state. In that case, just report any attribute
        # as not found to avoid an infinite recursion.
        if "f" not in self.__dict__:
            raise AttributeError()

        # Hack so that hydra.main can access the __code__ attribute of f and determine
        # the calling file
        return getattr(self.f, attr)
