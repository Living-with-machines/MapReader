# MapReader

<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-8-orange.svg?style=flat-square)](#contributors)
![PyPI](https://img.shields.io/pypi/v/MapReader)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Integration Tests badge](https://github.com/Living-with-machines/MapReader/actions/workflows/mr_ci.yml/badge.svg)
![DOI](https://zenodo.org/badge/430661738.svg)
![CodeCov](https://codecov.io/github/Living-with-machines/MapReader/graph/badge.svg?token=38GQ3O1GB5)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.06434/status.svg)](https://doi.org/10.21105/joss.06434)

## Table of Contents

- [MapReader](#mapreader)
  - [Table of Contents](#table-of-contents)
  - [What is MapReader?](#what-is-mapreader)
  - [Overview](#overview)
    - [MapReader classification pipeline](#mapreader-classification-pipeline)
    - [MapReader text spotting pipeline](#mapreader-classification-pipeline)
  - [Documentation](#documentation)
  - [What is included in this repo?](#what-is-included-in-this-repo)
  - [How to cite MapReader](#how-to-cite-mapreader)
  - [Acknowledgements](#acknowledgements)
  - [Contributors](#contributors)

---
<!--- sphinx-include --->

## What is MapReader?

**MapReader is a open-source python library for exploring and analyzing map images at scale.**

It contains two different pipelines:

- Classification pipeline: This pipeline enables users to fine-tune a classification model and predict the labels of patches created from a parent image.
- Text spotting pipeline: This pipeline enables users to detect and recognize text in map images.

MapReader was developed in the [Living with Machines](https://livingwithmachines.ac.uk/) project to analyze large collections of historical maps but is a _**generalizable**_ computer vision tool which can be applied to _**any images**_ in a wide variety of domains.

## Overview

### MapReader classification pipeline

The MapReader classification pipeline enables users to train a classification model to recognize visual features within map images and to identify patches containing these features across entire map collections:

<figure align="center">
  <img src="https://raw.githubusercontent.com/maps-as-data/MapReader/main/docs/source/_static/pipeline_explained.png"
        alt="MapReader pipeline"
        width="70%">
</figure>

### MapReader text spotting pipeline

The MapReader text spotting pipeline enables users to detect and recognize text in map images using a pre-trained text spotting model:

<figure align="center">
  <img src="https://raw.githubusercontent.com/maps-as-data/MapReader/main/docs/source/_static/text-spotting-pipeline.png"
        alt="MapReader text spotting pipeline"
        width="70%">
</figure>

## Documentation

The MapReader documentation can be found at https://mapreader.readthedocs.io/en/latest/.

**New users** should refer to the [Installation instructions](https://mapreader.readthedocs.io/en/latest/getting-started/installation-instructions/index.html) and [Input guidance](https://mapreader.readthedocs.io/en/latest/using-mapreader/input-guidance/) for help with the initial set up of MapReader.

**All users** should refer to our [User Guide](https://mapreader.readthedocs.io/en/latest/using-mapreader/) for guidance on how to use MapReader.
This contains end-to-end instructions on how to use the MapReader pipeline.

 **Developers and contributors** may also want to refer to the [API documentation](https://mapreader.readthedocs.io/en/latest/in-depth-resources/api/mapreader/) and [Contribution guide](https://mapreader.readthedocs.io/en/latest/community-and-contributions/contribution-guide/) for guidance on how to contribute to the MapReader package.


## Stay in touch

**All users** are encouraged to join our community! Please refer to the [Community and contributions](https://mapreader.readthedocs.io/en/latest/community-and-contributions/events.html) page for information on ways to get involved.

**Join our Slack workspace!**
Please fill out [this form](https://forms.gle/dXjECHZQkwrZ3Xpt9) to receive an invitation to the Slack workspace.

## What is included in this repo?

This repository contains everything needed for running MapReader.

The repository is structured as follows:

- `mapreader/`: Contains the source code for the MapReader library.
- `docs/`: Contains the documentation for the MapReader library.
- `tests/` and `test_text_spotting/`: Contains the tests for the MapReader library.
- `worked_examples/`: Contains worked examples of how to use the MapReader library.

## How to cite MapReader

If you use MapReader in your work, please cite:
- Our [JOSS paper](https://doi.org/10.21105/joss.06434) - to acknowledge the software irrespective of the version you used.
- Our [Zenodo record](https://zenodo.org/records/13643609) - to acknowledge the specific version of the software you used (or use the "Cite all versions?" option if your specific version isn't there).
- Optionally, our [SIGSPATIAL paper](https://dl.acm.org/doi/10.1145/3557919.3565812) to acknowledge the development of the software and the research behind it.

<!-- Add bibtext entries for the papers here -->

## Acknowledgements

This work was supported by Living with Machines (AHRC grant AH/S01179X/1), Data/Culture (AHRC grant AH/Y00745X/1) and The Alan Turing Institute (EPSRC grant EP/N510129/1).

Living with Machines, funded by the UK Research and Innovation (UKRI) Strategic Priority Fund, is a multidisciplinary collaboration delivered by the Arts and Humanities Research Council (AHRC), with The Alan Turing Institute, the British Library and the Universities of Cambridge, East Anglia, Exeter, and Queen Mary University of London.

Maps above reproduced with the permission of the National Library of Scotland https://maps.nls.uk/index.html

## Contributors

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://www.turing.ac.uk/people/researchers/katherine-mcdonough"><img src="https://avatars.githubusercontent.com/u/20363927?v=4?s=100" width="100px;" alt="Katie McDonough"/><br /><sub><b>Katie McDonough</b></sub></a><br /><a href="#research-kmcdono2" title="Research">🔬</a> <a href="#ideas-kmcdono2" title="Ideas, Planning, & Feedback">🤔</a> <a href="https://github.com/Living-with-machines/MapReader/commits?author=kmcdono2" title="Documentation">📖</a> <a href="#eventOrganizing-kmcdono2" title="Event Organizing">📋</a> <a href="#projectManagement-kmcdono2" title="Project Management">📆</a> <a href="https://github.com/Living-with-machines/MapReader/pulls?q=is%3Apr+reviewed-by%3Akmcdono2" title="Reviewed Pull Requests">👀</a> <a href="#talk-kmcdono2" title="Talks">📢</a> <a href="#tutorial-kmcdono2" title="Tutorials">✅</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://danielwilson.info"><img src="https://avatars.githubusercontent.com/u/34318222?v=4?s=100" width="100px;" alt="Daniel C.S. Wilson"/><br /><sub><b>Daniel C.S. Wilson</b></sub></a><br /><a href="#research-dcsw2" title="Research">🔬</a> <a href="#ideas-dcsw2" title="Ideas, Planning, & Feedback">🤔</a> <a href="#talk-dcsw2" title="Talks">📢</a> <a href="https://github.com/Living-with-machines/MapReader/commits?author=dcsw2" title="Documentation">📖</a> <a href="#eventOrganizing-dcsw2" title="Event Organizing">📋</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/kasra-hosseini"><img src="https://avatars.githubusercontent.com/u/1899856?v=4?s=100" width="100px;" alt="Kasra Hosseini"/><br /><sub><b>Kasra Hosseini</b></sub></a><br /><a href="https://github.com/Living-with-machines/MapReader/commits?author=kasra-hosseini" title="Code">💻</a> <a href="#ideas-kasra-hosseini" title="Ideas, Planning, & Feedback">🤔</a> <a href="#research-kasra-hosseini" title="Research">🔬</a> <a href="https://github.com/Living-with-machines/MapReader/pulls?q=is%3Apr+reviewed-by%3Akasra-hosseini" title="Reviewed Pull Requests">👀</a> <a href="#talk-kasra-hosseini" title="Talks">📢</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/rwood-97"><img src="https://avatars.githubusercontent.com/u/72076688?v=4?s=100" width="100px;" alt="Rosie Wood"/><br /><sub><b>Rosie Wood</b></sub></a><br /><a href="https://github.com/Living-with-machines/MapReader/commits?author=rwood-97" title="Code">💻</a> <a href="https://github.com/Living-with-machines/MapReader/commits?author=rwood-97" title="Documentation">📖</a> <a href="#ideas-rwood-97" title="Ideas, Planning, & Feedback">🤔</a> <a href="#talk-rwood-97" title="Talks">📢</a> <a href="#tutorial-rwood-97" title="Tutorials">✅</a> <a href="https://github.com/Living-with-machines/MapReader/pulls?q=is%3Apr+reviewed-by%3Arwood-97" title="Reviewed Pull Requests">👀</a> <a href="#maintenance-rwood-97" title="Maintenance">🚧</a> <a href="#research-rwood-97" title="Research">🔬</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://www.westerling.nu"><img src="https://avatars.githubusercontent.com/u/7298727?v=4?s=100" width="100px;" alt="Kalle Westerling"/><br /><sub><b>Kalle Westerling</b></sub></a><br /><a href="https://github.com/Living-with-machines/MapReader/commits?author=kallewesterling" title="Code">💻</a> <a href="https://github.com/Living-with-machines/MapReader/commits?author=kallewesterling" title="Documentation">📖</a> <a href="#maintenance-kallewesterling" title="Maintenance">🚧</a> <a href="https://github.com/Living-with-machines/MapReader/pulls?q=is%3Apr+reviewed-by%3Akallewesterling" title="Reviewed Pull Requests">👀</a> <a href="#talk-kallewesterling" title="Talks">📢</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://maps.nls.uk"><img src="https://avatars.githubusercontent.com/u/3666702?v=4?s=100" width="100px;" alt="Chris Fleet"/><br /><sub><b>Chris Fleet</b></sub></a><br /><a href="#data-ChrisFleet" title="Data">🔣</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/kasparvonbeelen"><img src="https://avatars.githubusercontent.com/u/11618160?v=4?s=100" width="100px;" alt="Kaspar Beelen"/><br /><sub><b>Kaspar Beelen</b></sub></a><br /><a href="#ideas-kasparvonbeelen" title="Ideas, Planning, & Feedback">🤔</a> <a href="https://github.com/Living-with-machines/MapReader/pulls?q=is%3Apr+reviewed-by%3Akasparvonbeelen" title="Reviewed Pull Requests">👀</a> <a href="#research-kasparvonbeelen" title="Research">🔬</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/andrewphilipsmith"><img src="https://avatars.githubusercontent.com/u/5346065?v=4?s=100" width="100px;" alt="Andy Smith"/><br /><sub><b>Andy Smith</b></sub></a><br /><a href="https://github.com/Living-with-machines/MapReader/commits?author=andrewphilipsmith" title="Code">💻</a> <a href="https://github.com/Living-with-machines/MapReader/commits?author=andrewphilipsmith" title="Documentation">📖</a> <a href="#mentoring-andrewphilipsmith" title="Mentoring">🧑‍🏫</a> <a href="https://github.com/Living-with-machines/MapReader/pulls?q=is%3Apr+reviewed-by%3Aandrewphilipsmith" title="Reviewed Pull Requests">👀</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->
