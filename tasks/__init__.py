"""Invoke tasks for the project.

Each module is exposed as a namespace, e.g. ``invoke download.historical``.
List every available task with ``invoke --list``.
"""

# %% IMPORTS

from invoke.collection import Collection

from tasks import dashboards, dev, download, predict

# %% NAMESPACES

ns = Collection()
ns.add_collection(Collection.from_module(download))
ns.add_collection(Collection.from_module(dashboards))
ns.add_collection(Collection.from_module(predict))
ns.add_collection(Collection.from_module(dev))
