import lightning
import nose
import os

lightning_dir = os.path.dirname(
    os.path.abspath(lightning.__file__))
os.chdir(lightning_dir)
nose.run(module=lightning)
