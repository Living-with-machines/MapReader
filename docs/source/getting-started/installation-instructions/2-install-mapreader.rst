Step 2: Install MapReader
==========================

Method 1: Install from `PyPI <https://pypi.org/project/mapreader/>`_
--------------------------------------------------------------------

If you want to use the latest stable release of MapReader and do not want/need access to the worked examples or MapReader code, we recommend installing from PyPI.
This is probably the easiest way to install MapReader.

-  Install ``mapreader``:

   .. code-block:: bash

      pip install mapreader

.. note:: To install the dev dependencies too use ``pip install "mapreader[dev]"`` instead.

To install the text-spotting dependencies, head to the :doc:`Spot text </using-mapreader/step-by-step-guide/6-spot-text>` section of the user guide!

Method 2: Install from source
-----------------------------

If you want to keep up with the latest changes to MapReader, or want/need easy access to the worked examples or MapReader code, we reccommend installing from source.
This method will create a ``MapReader`` directory on your machine which will contain all the MapReader code, docs and worked examples.

.. note:: You will need to have `git <https://git-scm.com/>`__ installed to use this method. If you are using conda, this can be done by running ``conda install git``. Otherwise, you should install git by following the instructions on `their website <https://git-scm.com/book/en/v2/Getting-Started-Installing-Git>`__.

-  Clone the ``mapreader`` source code from the `MapReader GitHub repository <https://github.com/Living-with-machines/MapReader>`_:

   .. code-block:: bash

      git clone https://github.com/Living-with-machines/MapReader.git

-  Install ``mapreader``:

   .. code-block:: bash

      cd MapReader
      pip install -v -e .

.. note:: To install the dev dependencies too use ``pip install -v -e ".[dev]"`` instead.

..
   Method 3: Install via conda (**EXPERIMENTAL**)
   ----------------------------------------------

   If neither of the above methods work, you can try installing MapReader using conda.
   This method is still in development so should be avoided for now.

   - Install MapReader directly from the conda package:

   .. code:: bash

      conda install -c anothersmith -c conda-forge -c defaults --override-channels --strict-channel-priority mapreader

   .. note:: The conda package seems to be sensitive to the precise priority of the conda channels, hence the use of the `--override-channels --strict-channel-priority` switches is required for this to work. Until this is resolve this installation method will be marked "experimental".
