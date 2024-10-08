How to add to or update the MapReader code
===========================================

MapReader's code is written in `Python <https://www.python.org/>`_.
There are a number of ways in which you can contribute to our code, these include:

- Reviewing part of the code.
- Adding new features.
- Fixing bugs.
- Adding clarity to docstrings.

Before you begin
-----------------

If you already have an idea for changes to the MapReader code, please ensure your idea has a corresponding issue on the `MapReader repository <https://github.com/Living-with-machines/MapReader>`_ (or create one if needed).
If, on the other hand, you would like to contribute to the code but don't currently have an idea of what to work on, head to our `open issues <https://github.com/Living-with-machines/MapReader/issues>`_ tab and find something that interests you.

In either case, before you begin working, either assign yourself to the issue or comment on it saying you will be working on making the required changes.

You will then need to fork the `MapReader repository <https://github.com/Living-with-machines/MapReader>`_ in order to make your changes.

Installation in development mode
--------------------------------

To ensure you have all the required dependencies for development, in your development environment, please install MapReader using the following commands:

.. code-block:: bash

    git clone https://github.com/Living-with-machines/MapReader.git
    cd MapReader
    pip install -e ".[dev]"

This will install MapReader in development mode, which means you can make changes to the code and see the effects without having to reinstall the package.

Style guide
-----------

When making your changes, please:

- Try to align to the `PEP 8 style guide for Python code <https://peps.python.org/pep-0008/>`__.
- Try to use the numpy-style docstrings (as per `this link <https://numpydoc.readthedocs.io/en/latest/format.html#>`__).
- Ensure all docstrings are kept up to date and reflect any changes to code functionality you have made.
- Add and run tests for your code.
- If you add new dependencies, add these to our ``setup.py``.
- If possible, update the MapReader user guide and worked examples to reflect your changes.

When you are done making changes, please:

- Run `black <https://black.readthedocs.io/en/stable/>`__ to reformat your code
- Run `flake8 <https://flake8.pycqa.org/en/latest/index.html#>`__ to lint your code.


Running tests
-------------

To run the tests, use the following command:

.. code-block:: bash

    python -m pytest .

This will run all the tests in the MapReader package.

If you've made changes to the MapReader code, please run the tests and ensure all are passing.
Or, if you've made code changes that intentionally alter the behavior of MapReader, please update the tests accordingly.

If possible, you should also add new tests for any new code/functionality you have added.

Updating documentation
----------------------

If possible, please update the MapReader documentation to reflect any changes you have made to the code.
You can see more guidance on this :doc:`here </community-and-contributions/contribution-guide/getting-started-with-contributions/add-or-update-documentation>`.

Updating the changelog
----------------------

If you have made changes to the code, please update the "Pre-release" section of the changelog (`CHANGELOG.md`) with a summary of your changes.

We follow the `Common Changelog <https://common-changelog.org/>`__ guidance for our changelog.
You can use their documentation and our previous changelog entries to help you format your own changelog entry.

.. note::

    If you have made only minor changes to the code (e.g. updating docstrings, minor style changes, formatting updates, etc.), you do not need to update the changelog. Instead you should add the `no changelog` label to your pull request.

When you are finished
----------------------

Once you are happy with the changes you have made, create a new `pull request <https://github.com/Living-with-machines/MapReader/pulls>`_ to let us know you'd like us to review your code.

If possible, please link your pull request to any issue(s) your changes fix/address and write a thorough description of the changes you have made.
