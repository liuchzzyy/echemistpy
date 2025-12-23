import pytest
from echemistpy.io.plugin_manager import get_plugin_manager, IOPluginManager


def test_plugin_manager_singleton():
    pm1 = get_plugin_manager()
    pm2 = get_plugin_manager()
    assert pm1 is pm2


def test_register_loader():
    pm = IOPluginManager()

    class MockLoader:
        pass

    pm.register_loader(["test", ".tst"], MockLoader)
    assert pm.get_loader("test") is MockLoader
    assert pm.get_loader(".test") is MockLoader
    assert pm.get_loader("TST") is MockLoader
    assert pm.get_loader(".tst") is MockLoader
    assert pm.get_loader("unknown") is None


def test_register_saver():
    pm = IOPluginManager()

    class MockSaver:
        pass

    pm.register_saver(["csv", "CSV"], MockSaver)
    assert pm.get_saver("csv") is MockSaver
    assert pm.get_saver("CSV") is MockSaver
    assert pm.get_saver("json") is None


def test_list_supported():
    pm = IOPluginManager()

    class MockLoader:
        pass

    pm.register_loader(["mpt"], MockLoader)
    assert ".mpt" in pm.list_supported_extensions()

    loaders = pm.get_supported_loaders()
    assert ".mpt" in loaders
    assert loaders[".mpt"] == "MockLoader"
