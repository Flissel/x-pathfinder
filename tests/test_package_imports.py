"""The package must be importable by its real name, from the repo itself."""


def test_package_imports_from_repo():
    from x_pathfinder.models import XAccount

    account = XAccount(handle="probe")
    assert account.handle == "probe"
    assert account.fitness_score == 0.0


def test_cli_entrypoint_is_importable():
    from x_pathfinder.cli import main

    assert callable(main)
