"""Register shared test options before test collection starts."""


def pytest_addoption(parser):
    parser.addoption(
        '--run-fitting-e2e',
        action='store_true',
        help='Run real scientific browser fits',
    )
