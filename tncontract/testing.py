"""
testing
==========

Function to run unit tests in tests/ directory using nose.

Functions named test_* in files named tests/test_* will be run automatically.
To run from command line use "nosetests -v tests" in tncontract/tncontract dir.
"""

def run():
    import nose
    # runs tests in qutip.tests module only
    nose.run(defaultTest="tncontract.tests", argv=['nosetests', '-v'])


if __name__ == "__main__":
    run()
