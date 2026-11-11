"""Invoke tasks for the project.

Each module is exposed as a namespace, e.g. ``invoke download.historical``.
List every available task with ``invoke --list``.
"""

# %% IMPORTS

from invoke.collection import Collection

from tasks import backtest, dashboards, dev, download, predict, update

# %% NAMESPACES

ns = Collection()
# Top-level task: `invoke update`
ns.add_task(update.update)
ns.add_collection(Collection.from_module(download))
ns.add_collection(Collection.from_module(dashboards))
ns.add_collection(Collection.from_module(predict))
ns.add_collection(Collection.from_module(backtest))
ns.add_collection(Collection.from_module(dev))
