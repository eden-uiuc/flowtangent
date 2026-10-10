import flowtangent as ft

def test_import():
    """Ensure the library imports cleanly and the core namespace is accessible."""
    # If the import above fails, the test fails automatically.
    # The assert just gives Pytest a concrete boolean to check.
    assert hasattr(ft, "Component")