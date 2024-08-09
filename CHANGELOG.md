# Changelog

## Python versions

The following table shows which versions of MapReader are compatible with which versions of python:

+----------+------------+
| Python   | MapReader  |
+==========+============+
| 3.9+     | v1.2.0     |
| 3.8-3.11 | <= v.1.1.5 |
+----------+------------+


## [v1.3.4](https://github.com/Living-with-machines/MapReader/releases/tag/v1.3.4) (2024-07-05)

<!--
PRs: #435
- adds new developments to the text spotting code:
    - Adds deduplicate methods for removing overlapping polygons within a patch and also between patches
    - Add overlap option when creating patches
    - Adds Runner base class to remove repeated code in the two different runners.
- fixes
    - #404: Add duplicate detection to text spotting code
    - #405: Add method for dealing with cut-off at patch edges in text spotting work

Commits:
add823a Merge pull request #435 from Living-with-machines/dev_text_spotting
1d76188 add test for overlap
6b2f282 fix ioa figure
a25251c update docs
1ee4fdf Missing M
2942935 add parent deduplication
9d67955 Removing placeholder text
e2bf35c Redoing list of community calls
169fea3 Editing upcoming communityevents
-->

-

## [v1.3.3](https://github.com/Living-with-machines/MapReader/releases/tag/v1.3.3) (2024-07-03)

<!--
PRs: #450, #449, #443, #438, #424
- Added sentence to related work to mention briefly that MapReader now includes text spotting.
- Addresses citation update request (openjournals/joss-reviews#6434 (comment)) for JOSS paper from @emdupre
- Update authors on paper.md
- Add a notebook for exploring and visualising text labels on maps for the June 2024 workshop in Lancaster
- Update workshop notebooks from feedback

cfc90e7 Merge pull request #450 from Living-with-machines/katie-paper-final-read
8cde900 rosie's edit
bd3d5f6 Merge pull request #449 from Living-with-machines/km-citation-update
382ec44 corcoran update
6d68559 paper update to mention text spotting
42ab95d update mapkurator citation
c6d078b Merge pull request #443 from Living-with-machines/rwood-97-patch-1
fa9a969 Update authors paper.md
c3cc3bd Merge branch 'main' into dev_text_spotting
6f19e41 Merge branch 'main' of https://github.com/Living-with-machines/MapReader
d2af7ab fix workshop notebook part3
5b5db89 update June date
8ad642d Merge pull request #438 from Living-with-machines/text-on-maps-viz
27420d2 clear outputs, filter for one parent map to save time, add a bit of text descriptions
2e6f997 Merge branch 'main' into text-on-maps-viz
1fd75c9 Merge pull request #424 from Living-with-machines/workshop_feedback
5040a2d updates from Katie feedback
011516c Update Events.rst
b0e04ec add reference for multilingual models
fd3ebda add link to NLS website
50d5a46 add text exploration notebook
816e6f9 update mapreader version
646f8f0 ensure compatibility with geopands 1.0.0a1 pre-release
b16a771 ignore workshop outputs
6f49ee0 update uk viz
b9784b1 small updates to plots
e5b8470 add data viz notebooks
cc73913 Add data/culture grant no. to paper
c238d9b add method for creating overlapping patches
733c483 rename workshops for 2024
f3d872e add deduplicate for parent images (e.g. if there is overlap between patches)
c65ff77 move common functions to base class
5d34bdf add deduplicate code to both runners
-->

-

## [v1.3.2](https://github.com/Living-with-machines/MapReader/releases/tag/v1.3.2) (2024-05-17)

<!--
PRs:

    Pull request 434
    ### Summary

    Adds calculation of average across all channels for mean/std pixel stats.
    Also makes sure image cropping is working for calc pixel stats.
    Fixes #426

    -------------------

    Pull request 433
    ### Summary

    Fixes #430

    -------------------

    Pull request 431
    ### Summary

    Fixes #64

    ### Describe your changes

    Command line call now prints mapreader version

    -------------------

    Pull request 432
    ### Summary

    Fixes #400

    ### Describe your changes

    Previous code allowed you enter a new label when reviewing labels but would give you an error (even though it relabelled the patch fine). Now we give an error and only allow user to enter one of the existing labels when re-labelling/reviewing their annotations.

    -------------------

    Pull request 418
    Addresses #386
    -------------------

    Pull request 415
    ### Summary

    Fixes non-geospatial README issues in #<386>

    ### Describe your changes

    Reformatted to standard fields per worked example, matching geospatial README.

    -------------------

    Pull request 414
    ### Summary

    Addresses "update geospatial README" in https://github.com/orgs/Living-with-machines/projects/7/views/1?pane=issue&itemId=59167565

    Fixes part of  #<386>

    ### Describe your changes

    Changed README from content that is now mostly in documentation site to clear structure for list of worked examples and required files.

    -------------------

    Pull request 416
    ### Summary

    Fixes #389

    ### Describe your changes

    - updated all typos mentioned in https://github.com/openjournals/joss-reviews/issues/6434#issuecomment-2041626766

    -------------------

    Pull request 396
    ### Summary

    Fixes #385

    ### Describe your changes

    Update code and dependencies to ensure fix warnings in tests

    To do:
    - [x] Update docs re. how to run tests
    - [x] Deal with annotator
    - [ ] Zero division error ?

    -------------------

    Pull request 413
    ### Summary

    Fixes #398

    ### Describe your changes

    e.g. https://maps.nls.uk/view/101437802 goes over the boundary and so is registered twice in the NLS metadata but with two sets of coordinates.
    Instead of erroring or overwriting the sheet, you will now get map_xxxx.png and map_xxxx_1.png saved and these will have separate metadata so separate coordinates.

    This also adds a mock function when downloading maps and so fixes #183

    -------------------

282b873 Merge pull request #434 from Living-with-machines/426-calc-pixel-stats
0a69fae update pixel stats calculation
ed2af0b Merge pull request #433 from Living-with-machines/430-add-id
5c63c92 fix add_id bug
20bdbd0 Merge pull request #431 from Living-with-machines/64-commandline
db75647 Merge pull request #432 from Living-with-machines/400-review-labels
ca42977 fix review_labels
2b3b985 fix command line script
b58331c add f-scores per class to docs
caa6547 add printing f-scores per class
acd9a7f fix typo
9dcb208 fix typo
d500aac add device argument to docs
3d0aae0 update y-labels in metric plots
b7c9981 update test instructions
eb00522 updates csv/tsv
23ab221 update YOUR_TURNs to align with docs
4a7a822 udpate notebooks
c4e9bc4 Merge pull request #418 from Living-with-machines/rw_docs
52643db Merge pull request #415 from Living-with-machines/non-geospatial_readme
5c165f7 remove annotations col from readme
c57c9f9 update example naming
6d2f6c0 Merge pull request #414 from Living-with-machines/geospatial_readme
b9c029d update instructions for worked examples
2e2e8e9 update geospatial readme table
cd64da3 remove postproc worked example
788c542 move annotation worked examples
a95800d Merge pull request #416 from Living-with-machines/documentation_typos
bc08df4 Merge pull request #396 from Living-with-machines/dev_tests
6aa535b Update Download.rst
e19fd31 Update Worked-examples.rst
bdfbb62 Update Worked-examples.rst
f4dcf6d Update Worked-examples.rst
2b336eb Update README.md
b39f320 Update README.md
fddd668 Update README.md
96723b0 Update non-geospatial README.md
a919af3 Update README.md
e94dae3 Update README.md
842fd39 Update README.md
46b4be3 Update README.md
dfb5c2a Update README.md
863178d add how to run tests
6e98c6e new README structure
ed427cd update annotator to fix warnings
437d67c Merge branch 'main' into dev_tests
fe370ee Merge pull request #413 from Living-with-machines/issue-398
04248fc remove re.search
-->

-

## [v1.3.1](https://github.com/Living-with-machines/MapReader/releases/tag/v1.3.1) (2024-05-03)

<!--
PRs:
    Pull request 399
    ### Summary

    Adding notebooks for the April/May workshop.


3d7ab10 update tests
dd216eb fix overwriting issue
1eb5aeb update saving for duplicate sheet names
565d8aa Merge pull request #399 from Living-with-machines/april_workshop
7f9cb6f update deepsolo notebook
08fab3a update version
e81c617 Merge branch 'main' into april_workshop
-->

-

## [v1.3.0](https://github.com/Living-with-machines/MapReader/releases/tag/v1.3.0) (2024-04-29)

<!--
PRs:

    Pull request 388
    ### Summary

    This PR begins the work of bringing text detection/recognition (ie. text spotting) into MapReader.

    Addresses #358

    ### Describe your changes

    `DeepSoloRunner` is a class used for running the [DeepSolo model](https://github.com/ViTAE-Transformer/DeepSolo/tree/main) on images. This produces polygons + text.
    `DPTextDETRRunner` is a class used for running the [DPText-DETR model](https://github.com/ymy-k/DPText-DETR) on images.  This produces just polygons.

    To do:

    - [x] Allow user to run on all patches in provided patch df
    - [x] Add plotting function
    - [x] Add docs for getting this running
    - [x] Check licences
    - [x] Add worked example
    - [ ] Add polygon merging code (?)

    -------------------

    Pull request 395
    ### Summary

    -------------------

    Pull request 384
    ### Summary

    Addresses #229
    Fixes #364

    ### Describe your changes

    This PR:
    - Updates supported python versions in MapReader from 3.8-3.11 to 3.9+
    - Adds cartopy as required dependency vs optional as 0.22 is simpler to install
    - Updates docs accordingly
    - Adds testing for python 3.12

    -------------------

f256ae9 Merge pull request #388 from Living-with-machines/ocr
6237341 update where to find configs and weights
197e4dd update docs for install
ce88217 update dptext detr runner import
af065a9 update filepath
fe02999 update install docs to use 3.1-0
2789f27 load 1000 pixel patdches
5f250c7 change other patches to 1000 meters
e874641 add metadata notebook
d6ab9bc fix patches_to_geojson problem
d1edfc5 update part2
0e8f059 address comments in part1 and extra
186f219 Update README.md
2d3584e Update README.md
b19a4cd add deepsolo notebook
c174515 fix errors
41568f6 add file to user guide toc tree
ad0f59c add docs for spotting text
867b83a update file saving docs
d7caa8d rename text_spot to spot_text
43a5b29 split notebooks and add inference only
a23fcb8 update notebook
fb2a780 add 2024 workshop notebook
3d69972 add version info to june 2023 notebook
9395917 update timm model names
62ba90f fix post processing tests
7ea1405 fix model weights warning
973a74a update minimum joblib
ba16e16 update post_process.py
30b2d1e update deps
9fb78fe add build to git ignore
e5d2312 Merge pull request #395 from Living-with-machines/kmcdono2-patch-1
7c4ad56 April community call update
672c762 update worked examples
5bf20f9 fix imports, add run_all arg to dptext detr runner
67b21b4 fix init
1950a67 rename to allow for different runners
2e2e2ba add worked example
6d8635b allow pass on import of DeepSoloRunner
7e193fc add to imports
680ceb1 add show method
9288e87 fix typo in show
043b132 Update publish-to-test-pypi.yml
4123a71 Merge branch 'main' into ocr
f3b151c Merge pull request #384 from Living-with-machines/dev_dependencies
a8194dc update for if patch_df not passed
c46b45e add run all method
e1fbd4e add deepsolo runner
f42b1c0 update installation instructions
3152729 update changelog with python version table
6bfd363 add changelog
-->

-

## [v1.2.0](https://github.com/Living-with-machines/MapReader/releases/tag/v1.2.0) (2024-04-08)

<!--
PRs:

    Pull request 383
    Updates the requirements on [flake8](https://github.com/pycqa/flake8) to permit the latest version.

    -------------------

    Pull request 380
    Updates the requirements on [pytest-cov](https://github.com/pytest-dev/pytest-cov) to permit the latest version.

    -------------------

    Pull request 381
    Updates the requirements on [black](https://github.com/psf/black) to permit the latest version.

    -------------------

    Pull request 382
    Updates the requirements on [torchvision](https://github.com/pytorch/vision) to permit the latest version.

    -------------------

    Pull request 379
    Updates the requirements on [pytest](https://github.com/pytest-dev/pytest) to permit the latest version.


e9d119f update docs
5a62b69 move cartopy to required and update version
d5ab0b8 force int for randint
4313c31 remove cap on torch version
a61defb add dependabot review workflow
16ac602 update python version in files
fab54f6 update allowed python versions
3e49c2b Merge pull request #383 from Living-with-machines/dependabot/pip/flake8-gte-6.0.0-and-lt-8.0.0
5b927b4 Merge pull request #380 from Living-with-machines/dependabot/pip/pytest-cov-gte-4.1.0-and-lt-6.0.0
312ebfd Update pytest-cov requirement from <5.0.0,>=4.1.0 to >=4.1.0,<6.0.0
30901b6 Update flake8 requirement from <7.0.0,>=6.0.0 to >=6.0.0,<8.0.0
e19bb02 Merge pull request #381 from Living-with-machines/dependabot/pip/black-gte-23.7.0-and-lt-25.0.0
86d92e2 Merge pull request #382 from Living-with-machines/dependabot/pip/torchvision-gte-0.11.1-and-lt-0.17.3
c37787c Merge pull request #379 from Living-with-machines/dependabot/pip/pytest-lt-9.0.0
a9705d3 Update dependabot.yml
6c21953 Update dependabot.yml
6e4397c Update torchvision requirement from <0.12.1,>=0.11.1 to >=0.11.1,<0.17.3
4bd8daf Update black requirement from <24.0.0,>=23.7.0 to >=23.7.0,<25.0.0
a284f4e Update pytest requirement from <8.0.0 to <9.0.0
20023aa Create dependabot.yml
-->

- Remove support for python 3.8
- Update dependencies and add cartopy as required dependency

## [v1.1.5](https://github.com/Living-with-machines/MapReader/releases/tag/v1.1.5) (2024-04-04)

<!--
PRs:

    Pull request 378
    ### Summary

    As per https://github.com/openjournals/joss-reviews/issues/6434 we are updating author list.

    ### Describe your changes

    Reorder authors

    -------------------

    Pull request 377
    ### Summary

    Addresses #364

    ### Describe your changes

    - Update supported python versions in docs
    - Update setup.py to be 3.8+

    -------------------

    Pull request 374
    ### Summary

    Thanks to @DGalexander for flagging this.

    This PR fixes mapreader's reproject coords functions when converting between coordinate systems.

    ### Describe your changes

    - coords are transformed from bounds:  (left, bottom, right, top) == (xmin, ymin, xmax, ymax)
    - always_xy is set to True to ensure no variation between crs ordering

    -------------------

    Pull request 373
    ### Summary

    In v1.1.3 I removed the `square_cuts` argument when patchifying images.

    In v1.1.2 or less, if you set `square_cuts=True`, when you reached an edge of an image, if your remaining 'slice' was less than the patch size you'd specified, you would move back a bit to create a square patch with some overlap with the previous patch. This essentially ensured all patches were square.
    In v1.1.3+, I removed this in favour of padding the patches at edges. This still means you end up with square patches but if your 'slice' was less than the patch size you pad with x no. of pixels to create a square patch.

    This PR re-adds the square-cuts argument to ensure bakcwards compatibility but with a deprecation warning.

    ### Describe your changes

    Add `_patchify_by_pixel_square()` method for the square cuts. This is just copy-paste of previous code and gets called if you set `square_cuts=True`.

    -------------------

    Pull request 372
    ### Summary

    This PR:
    - updates the link to contribution guide in the README
    - adds info about installing MapReader in development mode
    - adds info about how to run MapReader tests

    Fixes #371
    Addresses #369

    -------------------

    Pull request 366
    ### Summary

    This removes the conda install instructions from the docs since it is outdated.
    Addresses #365
    Addresses #364

    We should try to fix install asap as per #162

    -------------------

    Pull request 363
    ### Summary

    This allows users to specify whether to download in parallel vs sequentially.
    Sequentially is slower but can overcome problems to do with rate limits when downloading from a tileserver.

    ### Describe your changes

    Feed the download_in_parallel argument up to download methods.

    -------------------

    Pull request 362

    -------------------


75e2824 Merge pull request #378 from Living-with-machines/JOSS_paper
0a61a89 Merge pull request #377 from Living-with-machines/rw_docs
c1b9223 Update Install.rst
be8a63f Update author list
0389a25 Update cartopy instructions
a65db27 add commas
5e73700 Update setup.py - add cartopy
906e77b Update setup.py
c762f42 Update supported python versions
de910b4 Merge pull request #374 from Living-with-machines/coords_bug_fix
13e0f88 Merge pull request #373 from Living-with-machines/dev_load
7d1b682 add tests
a79293f unsupress decompression bomb error
42bb6dd fix transform
1c3d5f3 readd square cuts option
ef730c8 supress decompression bomb error
7d0bfa6 add pyogrio to dependencies
141fa00 Merge pull request #372 from Living-with-machines/rw_docs
3432f08 add info about dev environment and tests
7d3f191 fix link to contribution guide
32e8b91 Merge pull request #366 from Living-with-machines/rw_docs
a1277bb comment out conda install
e99bd6e allow users to specify file names in downloader
54db894 Merge pull request #363 from Living-with-machines/dev_download
272d84e community calls
734e467 Merge pull request #362 from Living-with-machines/kmcdono2-patch-1
03aca08 Merge branch 'main' into kmcdono2-patch-1
aa51ed6 Update Events.rst
19f66ab more fix lists
09f059c fix lists
94c6763 fix lists
f106c95 Update Events.rst
6452513 allow users to specify whether to download in parallel
-->

-

## [v1.1.4](https://github.com/Living-with-machines/MapReader/releases/tag/v1.1.4) (2024-02-23)

<!--
PRs:
    Pull request 357
    ### Summary

    Our worked examples were all slightly outdated and so some users were having trouble when running them.
    This PR updates them to work with MapReader v1.1.4.

    Fixes #355

    ### Describe your changes

    - Fix annotator for if you don't have urls
    - Only return important columns
    - Only add to parents if parents is not None


4ae6026 Merge pull request #357 from Living-with-machines/rw_docs
1d1ca54 Update Worked-examples.rst
deeb7cb update docs
f6ecb19 update mnist notebook
133d0b6 add readme for workshop notebooks
2c105e0 update context notebook
35e3142 update geospatial pipeline
02418ee fix plants worked example
c837608 allow for patches with no parent
ee4e92f update annotate worked examples
-->

-

## [v1.1.3](https://github.com/Living-with-machines/MapReader/releases/tag/v1.1.3) (2024-02-21)

<!--
PRs:

    Pull request 350
    ### Summary

    At the moment, MapReader has the option to use the context (i.e. central patch + surrounding patches) to improve the accuracy of the model. This is done with a custom model which has two branches and two separate input pipelines (one for patch only, one for context image). However, the context model code doesn't work very well.

    An idea to fix/improve this is to use a single branch model but use only the context image as input. Annotations will be at the patch level so the model should hopefully learn that annotations correspond to patterns in the centre of the context image (i.e. central patch) rather than the whole image.

    Addresses #17
    Addresses #287

    ### Describe your changes

    - `square_cuts` argument is removed as option from `images.py` (Load subpackage). Patches at edges are now padded to create square patches so patch size is consistent across entire parent image.
    - `annotator.py` now builds context image based on size of current patch so should work for patchify by meters
    - `context_dataset` argument added when creating datasets in `load_annotations.py`. This calls a `create_context_datasets` method which creates a context dataset vs patch dataset.
    - Context images in `PatchContextDataset` are now created by loading surrounding patches (as opposed to by no. of pixels either side). This is consistent with previous steps in the pipeline.
    - `PatchContextDataset` now returns ONLY context image (before it was context image + patch image).
    - `classifier.py` now works for context datasets and patch datasets (`context_classifier.py` is deleted as now redundant)
    - Updated docs

    -------------------

    Pull request 356
    ### Summary

    Bug fix for annotations since `filter_for` argument added.

    ### Describe your changes

    I made it so filtering doesn't happen if no `filter_for` is given.

    -------------------

    Pull request 354
    ### Summary

    Update paper as per requests - https://github.com/openjournals/joss-reviews/issues/6168

    Fixes #353

    -------------------

    Pull request 319
    ### Summary

    Since we have run the 2nd ed model on a set of 1st ed patches, we need a method of looking at the predicted labels and marking them as correct/incorrect. It would be good if we could do this using the new annotator.
    This relates to [this issue](https://github.com/Living-with-machines/railspace/issues/1).

    ### Describe your changes

    This PR:
    - Adds `filter_for` argument to the annotator class. You can now pass a dict (e.g. ``{"predicted_label":"railspace"}``) to the annotator and it will filter your patch_df for this col and value.

    For annotating predictions, I guess we just have labels of "correct" "incorrect" and "unsure" and then we can annotate as normal but using this filter for each label.

    Theres probably more to do - will think.

    -------------------

    Pull request 342
    ### Summary

    As per #339, this PR implements a post-processing script so that users can filter out false positives.
    This works for linear features or anything where you expect multiple patches to be clustered but solo patches would be false positive.
    It also adds a `save_predictions()` method the classifier to make sure predictions and confidence scores are saved in format expected for post-processing.

    Fixes #218
    Addresses #339

    -------------------


a429260 Merge pull request #350 from Living-with-machines/context_classifier_single
113f3a1 better test for scramble frame
bf7c33b test backward compatibility
9e75034 ensure backwards compatibility
6bca80a only save important cols in annotator
b7121fe update docs
bf396cb add datasets tests and fix parhugin code
d22cad9 Merge branch 'main' into context_classifier_single
a61d4c6 add tests for datasets
05ad379 add tests for geotiff saving (edge patches)
7315b4f ensure pixel stats are correct for edge patches
23f1920 update test_classifier
430da59 fix test_annotations_loader
c1003af update/add tests
ac1b79c add worked example for context classification
cc530ef Merge pull request #356 from Living-with-machines/annotation_fix
cb823bf update notebook
46de344 fix filter for
12cf51a Merge pull request #354 from Living-with-machines/paper
426adbf fix typo
5a0347f Fix (?) references
0d1fb68 Update affiliations
7679101 update subtitle
3152716 update/fix tests
4593795 update sample annots file
0e1eef4 fix index map vs apply
a4c2687 Merge branch 'main' into context_classifier_single
cca2b15 fix typo
4349288 Merge pull request #319 from Living-with-machines/analyse_preds
0cd7c55 update docs
afd693d add tests
745e414 Merge branch 'main' into analyse_preds
033917f Merge pull request #342 from Living-with-machines/339-postproc
777c857 Merge branch 'main' into 339-postproc
6228aa2 Update codecov fail in CI
1abce20 add suggestion
-->

-

## [v1.1.2](https://github.com/Living-with-machines/MapReader/releases/tag/v1.1.2) (2024-02-05)

<!--
PRs:

    Pull request 374
    ### Summary

    Thanks to @DGalexander for flagging this.

    This PR fixes mapreader's reproject coords functions when converting between coordinate systems.

    ### Describe your changes

    - coords are transformed from bounds:  (left, bottom, right, top) == (xmin, ymin, xmax, ymax)
    - always_xy is set to True to ensure no variation between crs ordering

    -------------------

    Pull request 345
    ### Summary

    Update links in docs to be relative. This means that if you are looking at old docs then clicking a link won't take you to v. latest.

    Update rail_space and rail space to railspace.

    Fixes #320

    -------------------

    Pull request 338
    ### Summary

    This PR implements changes from PRs #318, #326 and #335 and #337.
    See specifics in these PRs and related issues.

    -------------------


    Pull request 318
    ### Summary

    At the moment, the code can output patches as geojson and geotiffs. These methods rely on coordinates being saved in the patch/parent metadata.

    Problems with this come from:
    - Coordinates are saved as a list in the patch/parent dataframes (`[minx, miny, maxx, maxy]`) so when reading from a csv, these coords are read as a string and not a python list object. This is a problem when indexing the coord list later on.
    - In an earlier version of mapreader, the parent coordinates we calculated incorrectly. If these coords are in your parent dataframe it is not possible to recalculate using other info saved for hte parent image (i.e. the grid bounding box used for downloading)
    - The save to geojson method uses a shapely polygon (generated from coordinates) for saving geometry instead of the coordinates. When reading from a csv, the polygon is read as a string not a polygon object.

    ### Describe your changes

    This PR:
    - Adds method to apply `literal_eval` to all columns in dataframe using try/except. This fixes problems with tuples/coordinates/etc being read as strings. i.e. fixes problem with coords list.
    - Adds method of regenerating coordinates from grid bounding boxes. i.e. to fix previously incorrect coords from bug in earlier version of MapReader.
    - Recreates shapely polygon from string if needed
    - Adds method for saving parent images as geotiffs. This is helpful because both parent/patch images can then be loaded into QGIS or other geospatial software and compared to a basemap to check that coords are as expected.

    -------------------

    Pull request 326
    ### Summary

    This PR adds tests for coordinate saving.
    These coords have been checked by saving maps as geotiffs and comparing in QGIS against an OSM basemap.

    Fixes #279

    This PR should be merged after #318 as it has taken some changes from there.
    `git merge main` before merging this PR.

    ### Describe your changes

    - Coords are tested for both zoom level 10 and 14.

    -------------------

    Pull request 335
    Fixes #331

    ### Summary

    This PR changes the calculation of image shape such that the height, width and no. of channels are taken explicitly from the image not the array shape. This means that single band images get channels =1 rather than no value for channels.

    ### Describe your changes

    - Change how shape is calculated
    - Update plotting to allow for single band images
    - Updates geotiff saving to allow for single band images

    -------------------

    Pull request 337
    ### Summary

    Fixes #332
    Fixes #333

    ### Describe your changes

    This PR:
    - removes kwargs from the Annotator's `__init__()` method. This means that if you spell something wrong or pass an unrecognised arg you will get an error.
    - removes error if URL not found in parent df.
    - resets index when loading annoations, this means if your annotations df has "image_id" in col 0 then it will not error.

    -------------------

f31d87f Merge pull request #347 from Living-with-machines/346-annotations-order
b52086e fix tests
74c4c2d fix tests for random order of annotations
e50aaa1 fix sortby
cf91abb fix queue
8ba4cb3 Merge pull request #345 from Living-with-machines/rw_docs
6c33f7b update docs
f668a73 Add post-processing docs
3c58460 add tests
9b9003c force image_id index
dc848c3 use total_df to build context images
d8a08e2 force image_id index
fe05f91 remove context annotations from annotator
e58acd2 return only context image for context datasets
6f2a882 keep all cols when saving
5cc37e7 only add context annotations to annotated patches
5d54f5e rename context dataset trasnforms for clarity
02d0e67 fix load annotations
f7baba7 use iloc not at for getting data
a71a34b allow users to annotate at context-level
84340b0 fix context for annotator
c1b596c ensure geotiffs are saved correctly
a1e7941 remove square_cuts arg from tests
428f0f3 update context saving
34014b1 return df after eval
e978b40 replace `square_cuts` with padding at edge patches
08136a4 skip edge patches, allow new labels
f6f5e89 add docstrings, allow user to specify conf
60641bb add post processing script
02e2436 enable easier saving of predictions to csv
5e796ce update delimiter
b936bed change delimiter
11afa54 Merge branch 'main' into context_classifier
566e602 Merge pull request #338 from Living-with-machines/dev
-->

-

## [v1.1.1](https://github.com/Living-with-machines/MapReader/releases/tag/v1.1.1) (2024-01-08)

<!--
PRs:

    Pull request 337
    ### Summary

    Fixes #332
    Fixes #333

    ### Describe your changes

    This PR:
    - removes kwargs from the Annotator's `__init__()` method. This means that if you spell something wrong or pass an unrecognised arg you will get an error.
    - removes error if URL not found in parent df.
    - resets index when loading annoations, this means if your annotations df has "image_id" in col 0 then it will not error.

    -------------------

    Pull request 326
    ### Summary

    This PR adds tests for coordinate saving.
    These coords have been checked by saving maps as geotiffs and comparing in QGIS against an OSM basemap.

    Fixes #279

    This PR should be merged after #318 as it has taken some changes from there.
    `git merge main` before merging this PR.

    ### Describe your changes

    - Coords are tested for both zoom level 10 and 14.

    -------------------

    Pull request 335
    Fixes #331

    ### Summary

    This PR changes the calculation of image shape such that the height, width and no. of channels are taken explicitly from the image not the array shape. This means that single band images get channels =1 rather than no value for channels.

    ### Describe your changes

    - Change how shape is calculated
    - Update plotting to allow for single band images
    - Updates geotiff saving to allow for single band images

    -------------------

    Pull request 318
    ### Summary

    At the moment, the code can output patches as geojson and geotiffs. These methods rely on coordinates being saved in the patch/parent metadata.

    Problems with this come from:
    - Coordinates are saved as a list in the patch/parent dataframes (`[minx, miny, maxx, maxy]`) so when reading from a csv, these coords are read as a string and not a python list object. This is a problem when indexing the coord list later on.
    - In an earlier version of mapreader, the parent coordinates we calculated incorrectly. If these coords are in your parent dataframe it is not possible to recalculate using other info saved for hte parent image (i.e. the grid bounding box used for downloading)
    - The save to geojson method uses a shapely polygon (generated from coordinates) for saving geometry instead of the coordinates. When reading from a csv, the polygon is read as a string not a polygon object.

    ### Describe your changes

    This PR:
    - Adds method to apply `literal_eval` to all columns in dataframe using try/except. This fixes problems with tuples/coordinates/etc being read as strings. i.e. fixes problem with coords list.
    - Adds method of regenerating coordinates from grid bounding boxes. i.e. to fix previously incorrect coords from bug in earlier version of MapReader.
    - Recreates shapely polygon from string if needed
    - Adds method for saving parent images as geotiffs. This is helpful because both parent/patch images can then be loaded into QGIS or other geospatial software and compared to a basemap to check that coords are as expected.

    -------------------

    Pull request 336
    Fixes links in the Classify Train/Infer pages.

    -------------------

    Pull request 330
    This PR adds the codecov badge to the repo.
    Fixes #314

    Note: Other badge formats are available - see https://app.codecov.io/gh/Living-with-machines/MapReader/settings/badge for info.

    -------------------

    Pull request 328
    To add codecov to the repo.
    Addresses #314

    - [ ] Add badge

    -------------------

    Pull request 316
    ### Summary

    This PR adds the JOSS paper and references file to the repo in a directory called `paper`.
    It also adds a GH action which builds a draft of the paper - we should turn this off once we are done making changes!

    Fixes #257

    ## To do still

    - [x] Check output of GH action
    - [x] Fill in title, AHRC grant no., anything else that is currently "XXXX".


8997b4d add missing tests
31be1d0 Merge pull request #337 from Living-with-machines/dev_annotator
a086c1f remove fail on no url col
00734f9 Merge branch 'dev' into dev_annotator
8cd9572 Merge pull request #326 from Living-with-machines/279-test-coord-saving
2efad70 Merge pull request #335 from Living-with-machines/331-hwc-bug
e02f857 add tests for grayscale images
a264b8c Merge branch 'dev' into 331-hwc-bug
3f8a36f Merge branch 'dev' into 279-test-coord-saving
7eb659e Merge pull request #318 from Living-with-machines/fix_save_to_geojson
af7df2d add saving of one band geotiffs
7e90e2b add ClassifierContainer imports to docs
932df44 allow for image_id to be column 0
6dac40f remove error if no url
3b2c139 remove kwargs
d8468e0 Merge pull request #336 from Living-with-machines/rw_docs
314bdd6 fix links
b8d3da4 calc shape from height, width and channels explicitly and allow for single band images
6e49e90 update docs
86a240b remove unnecessary literal_evals
8bda0c2 add more tests
73ed889 add and update tests
cb4fd4a update metadata files
8c6bb07 Merge branch 'main' into fix_save_to_geojson
2236f3e Merge pull request #330 from Living-with-machines/codecov_badge
861f9e5 Rename Contributors.md to contributors.md
4d8fccb add codecov badge
cce63d8 Merge pull request #328 from Living-with-machines/codecov
4fcaaae Update mr_ci.yml
006266d Merge pull request #316 from Living-with-machines/paper
130b969 update mr_ci.yml
3446bc4 update mr_ci.yml
1bf0625 Update mr_ci.yml
0f34d74 add notebook for how to annotate model predictions
dce2e59 add filter_for to docs
670137c Merge branch 'main' into analyse_preds
edf22d5 minor updates + v number
02c2984 add approx for coords
31cfbf9 add tests for coord saving (downloader)
-->

-

## [v1.1.0](https://github.com/Living-with-machines/MapReader/releases/tag/v1.1.0) (2023-12-14)

<!--
PRs:

    Pull request 322

    ### Summary

    Fixes #316 comment about Project CV being unclear (re: computer vision vs. curriculum vitae)

    ### Describe your changes

    Changed title in header from "Project CV" to "Project Curriculum Vitae"

    -------------------

    Pull request 321
    ### Summary

    Fixes @kasra-hosseini feedback in #316

    ### Describe your changes

    - add citation to JVC paper as reference for statement about map digitization
    - add comment about documentation in conclusion

    -------------------

    Pull request 173
    Attempting to create a new `Annotator` class for MapReader, which will provide a less buggy experience for the user who is trying to annotate their patches.

    Feature requests:
    - [X] ensure joining on existing annotations works
    - [X] ensure not re-annotating
    - [X] showing context
    - [x] Add in `sortby` keyword, for sorting by a different column (existed in prior version)
    - [x] Add a `min_mean_pixel` and `max_mean_pixel` for filtering data shown to you (existed in prior version)
    - [x] display URL to NLS map
    - [x] keep patch filepaths in annotations csv output
    - [x] keep label names (either instead of or in addition to label indices) in annotations csv output.
    - [X] margin
    - [x] "next random patch"
    - [ ] keyboard shortcuts
    - [ ] batch sizing
    - [ ] restrict to bounding box
    - [ ] consider `load` and `dump` method/s (for, for instance LabelStudio input/output etc.)
    - [x] Make it obvious that 'next' and 'previous' are next random patch not like moving left/right on context image

    -------------------


8eda75e Merge pull request #322 from Living-with-machines/kmcdono2-docs-fix
8dadad2 Merge pull request #321 from Living-with-machines/paper-katie-update
f090542 Merge remote-tracking branch 'origin/fix_save_to_geojson' into 279-test-coord-saving
f342c65 add printing of filter
ccbf46c Merge pull request #173 from Living-with-machines/kallewesterling/issue166
3734439 rename as "Project Curriculum Vitae"
78d78bd Update paper.md per Kasra's comments
5fdde2a Update docs/source/User-guide/Annotate.rst
7afb1c3 Update docs/source/User-guide/Annotate.rst
-->

-

## [v1.0.7](https://github.com/Living-with-machines/MapReader/releases/tag/v1.0.7) (2023-12-14)

<!--
PRs:

    Pull request 277
    `size_in_m`  variable was outside if statement and sometimes references before being set.

    Tests should be added as part of #279

    -------------------

    Pull request 317
    ### Summary

    Fixes #313

    -------------------

    Pull request 312
    Adds @andrewphilipsmith as a contributor for code, doc, mentoring, review.

    -------------------

    Pull request 311
    Adds @kallewesterling as a contributor for maintenance, review, talk.

    -------------------

    Pull request 310
    Adds @rwood-97 as a contributor for ideas, talk, tutorial, review, maintenance, research.

    -------------------

    Pull request 309
    Adds @kasra-hosseini as a contributor for ideas, research, review, talk.

    -------------------

    Pull request 308
    Adds @dcsw2 as a contributor for ideas, talk, doc, eventOrganizing.

    -------------------

    Pull request 307
    Adds @kmcdono2 as a contributor for doc, eventOrganizing, projectManagement, review, talk, tutorial.

    -------------------

    Pull request 306
    Adds @kasparvonbeelen as a contributor for ideas, review, research.

    -------------------

    Pull request 305
    Adds @ChrisFleet as a contributor for data.

    -------------------

    Pull request 304
    Adds @kallewesterling as a contributor for code, doc.

    -------------------

    Pull request 303
    ### Summary

    -------------------

    Pull request 302
    Adds @rwood-97 as a contributor for doc.

    -------------------

    Pull request 301
    Adds @kasra-hosseini as a contributor for code.

    -------------------

    Pull request 300
    Adds @dcsw2 as a contributor for research.

    -------------------

    Pull request 299
    Adds @kmcdono2 as a contributor for research, ideas.

    -------------------

    Pull request 298
    Adds @rwood-97 as a contributor for code.

    -------------------

    Pull request 297
    ### Summary

    Add contributors bot

    -------------------

    Pull request 290
    ### Summary

    -------------------


9ee0cdf Merge branch 'kallewesterling/issue166' into analyse_preds
462983d change how max_size is set in lieu of resize_to param
7369464 Merge branch 'kallewesterling/issue166' into analyse_preds
3ff08ca updates docs for resize_to
bb56cc6 add ``resize_to`` kwarg to resize small patches
621a208 minor
b152086 Add an example about 16K map sheets
c6ebab7 Small changes in the first paragraph; reordered tags and added DL
15460f6 add surrounding arg to docs
f11af85 minor update to annotator
04a314d add filter_for arg to annotator
e385370 Merge pull request #277 from Living-with-machines/fix_geo_utils
255c0c1 Merge pull request #317 from Living-with-machines/rw_docs
019c13f change false to none
a6a981d Merge branch 'main' into fix_geo_utils
979fb4b update docstrings
cca139c add tests for delimiter
7e19d9b add contributor docs
92b9a28 fix sorting
a4d2028 add tests and minor update to annotator.py
98e3eca add paper
1e7634d update test imports
4b259b0 add geopandas to dependencies
708716e Merge branch 'main' into kallewesterling/issue166
1e36624 update docs
df886c8 update notebook
5cd432e update setup.py to work with jupyter notebook/lab
0dd7082 fix sortby and min/max values
ddecc8a add literal_eval for reading list/tuple columns
14b8d29 add saving coords from grid_bb
b877a96 fix patch coords
2457d2e add kwargs as normal args (not tested)
ebf98ae add method to save parents as geotiffs
393089a add literal eval for list/tuple columns
53766b6 Update .all-contributorsrc
8972876 Merge pull request #312 from Living-with-machines/all-contributors/add-andrewphilipsmith
e8075cf docs: update .all-contributorsrc [skip ci]
33f95ca docs: update README.md [skip ci]
49cf5e9 Merge pull request #311 from Living-with-machines/all-contributors/add-kallewesterling
9c215ae docs: update .all-contributorsrc [skip ci]
de02c1e docs: update README.md [skip ci]
a9b68a9 Merge pull request #310 from Living-with-machines/all-contributors/add-rwood-97
501f070 docs: update .all-contributorsrc [skip ci]
d3964b0 docs: update README.md [skip ci]
54d9ff7 Merge pull request #309 from Living-with-machines/all-contributors/add-kasra-hosseini
af05661 docs: update .all-contributorsrc [skip ci]
aeeb468 docs: update README.md [skip ci]
88fb6a0 Merge pull request #308 from Living-with-machines/all-contributors/add-dcsw2
6132c78 docs: update .all-contributorsrc [skip ci]
abbae52 docs: update README.md [skip ci]
768f703 Merge pull request #307 from Living-with-machines/all-contributors/add-kmcdono2
ed12d0c docs: update .all-contributorsrc [skip ci]
678c8c2 docs: update README.md [skip ci]
643d711 Merge pull request #306 from Living-with-machines/all-contributors/add-kasparvonbeelen
79a8c1e docs: update .all-contributorsrc [skip ci]
7ddcd98 docs: update README.md [skip ci]
d0b5e02 Merge pull request #305 from Living-with-machines/all-contributors/add-ChrisFleet
60024ce docs: update .all-contributorsrc [skip ci]
958bcb9 docs: update README.md [skip ci]
742a6f5 fix save to geojson
054656a rename annotator file
e0950c9 update notebook
6a79739 Merge branch 'main' into kallewesterling/issue166
94c0346 Merge pull request #304 from Living-with-machines/all-contributors/add-kallewesterling
92f0a1a docs: update .all-contributorsrc [skip ci]
6edd0d5 docs: update README.md [skip ci]
a72bdcc Merge pull request #303 from Living-with-machines/rwood-97-patch-2
c9f7796 Update README.md
fdf3d6d Update README.md
e1fce5b Update .all-contributorsrc
581d421 Update README.md
da344b7 Update README.md
ed4e4cb Merge pull request #302 from Living-with-machines/all-contributors/add-rwood-97
42d23a3 Merge branch 'main' into all-contributors/add-rwood-97
2a86d1a Merge pull request #301 from Living-with-machines/all-contributors/add-kasra-hosseini
7664427 Merge branch 'main' into all-contributors/add-kasra-hosseini
849d0e6 Merge pull request #300 from Living-with-machines/all-contributors/add-dcsw2
22baae4 Merge branch 'main' into all-contributors/add-dcsw2
d41acec Merge pull request #299 from Living-with-machines/all-contributors/add-kmcdono2
0c65088 docs: update .all-contributorsrc [skip ci]
c08c6fb docs: update README.md [skip ci]
aecb5ee docs: update .all-contributorsrc [skip ci]
1b88b6d docs: update README.md [skip ci]
339cf78 docs: update .all-contributorsrc [skip ci]
3731d70 docs: update README.md [skip ci]
a80550d docs: update .all-contributorsrc [skip ci]
e56b36d docs: update README.md [skip ci]
d43cafe Merge pull request #298 from Living-with-machines/all-contributors/add-rwood-97
66651ee docs: create .all-contributorsrc [skip ci]
8ca5b51 docs: update README.md [skip ci]
a2b26c5 Merge pull request #297 from Living-with-machines/rwood-97-contributors
9ec8657 Create .all_contributors.rc
7a57fa2 Update README.md
c0a7278 Update README.md (add contributors)
0b33fa1 add docs on how to use context model
d8f31d3 remove context container from init imports
55304d5 remove classifier context (now all in one)
7f6610b add context option for generate_layerwise_lrs
bb0ec8d update confusing language in params2optimize
d798454 update attribute names in custom model for clarity
f83a5f3 process inputs as a tuple
5ac8f1e always return images as tuple
ef688d5 fix color printing
0b170cd Merge pull request #290 from Living-with-machines/rw_docs
38b1c65 update trainable_col arg name
ac48935 align classifier_context to classifier
46e0fe0 enable annotations loader to create patch context dataset
-->

-

## [v1.0.6](https://github.com/Living-with-machines/MapReader/releases/tag/v1.0.6) (2023-11-22)

<!--
PRs:



c10b2c4 Merge pull request #291 from Living-with-machines/dev_download
bc37c36 add docs
20f47c6 Update PULL_REQUEST_TEMPLATE.md
e881180 update geo pipeline test
6fb1a14 update/add tests
40624e9 add try/excpt to sheet downloader
369677e Allow user to select metadata to save
740314d update date saving for extract_published_dates
d512d4e update section headers
e2f2027 Merge pull request #280 from Living-with-machines/dev_classify
2acda4e Merge pull request #285 from Living-with-machines/dev_download
09fb4a3 run pre-commit
aa98ed8 Merge branch 'main' into dev_download
c0ceec8 fix drop duplicates
093f272 Update About.rst
dfdbd1f Merge pull request #283 from Living-with-machines/rw_docs
ac00727 update about docs
bbb2c8f fix typo
d99df36 add tests
4743698 Add docs for inference only
7674eb9 i actually tested it this time
8039c85 fix typo (fix tests)
bc4c0f0 fix tests
48dda3b fix file paths
a849a14 update docs - split into train/infer
eb396e4 fix adding of dataloaders if load_path also passed
386c5ed Merge branch 'main' into dev_classify
9ac99b2 update docs
7517378 fix notebooks
3b017a0 fix tests
2335e81 Merge pull request #278 from Living-with-machines/pre-commit
36512a9 Merge branch 'main' into dev_classify
e526e11 only require criterion for training/validation
4b52302 add default for dataloaders arg
a4e1e7a add option to load ClassifierContainer with no dataloaders
8443f4a run all
50c6c03 run pyupgrade
b1626ed add pyupgrade
7437f8c Merge branch 'main' into pre-commit
1a7a8f2 remove backslashes
26977d7 add create_dataloader method to PatchDataset
6678a22 add __init__.py and test_import.py to excludes
e0d80ca exclude worked examples from pre-commit
8fdb38d Merge pull request #276 from Living-with-machines/dev_download
32cdc9e Merge branch 'dev_download' of https://github.com/Living-with-machines/MapReader into dev_download
46e5688 fix coordinate saving
9a2cf34 Update Contributors.md
29d08d1 fix reproject geo info
6f1f96d only drop absolute duplicates
65204be Merge pull request #270 from Living-with-machines/dev_download
8a7320d ensure download doesn't fail if maps are not found
8f15ae5 Merge pull request #269 from Living-with-machines/dev_download
e852818 raise error is both corners are missing
71c0300 check both upper and lower corners when finding tilesize
690172a Update Contributors.md
7322570 Update README.md
d276c10 Update ways_of_working.md
4b9f787 Update Contributors.md
28e4cf7 pre-commit run all
d739d3e add pre-commit and ruff configs
b26041d Merge pull request #258 from Living-with-machines/dev_testing
71c5d22 Update images.py
377d480 add geo pipeline tests
cd612e4 Merge pull request #256 from Living-with-machines/rw_docs
a925df7 Create Contributors.md
66f533c Update ways_of_working.md
762e524 split Code of conduct
1bf2d24 add citation info
24e160a add DOI badge
a734257 Merge pull request #253 from Living-with-machines/rw_docs
20a0ccb fix typos
1c3f9a3 Update Project-cv.rst
006e1c8 Update Project-cv.rst
bb55cef katie updates
a956f2d Update Project-cv.rst
b30e5ce Update Project-cv.rst
a37c120 Update Project-cv.rst
aa70fd7 Update Project-cv.rst
c6bf97a Update Project-cv.rst
75be50c add project cv and events page
b1248a1 Update Install.rst
c6ffda6 Merge pull request #246 from Living-with-machines/162-fix-conda-deploy
c85646f Merge branch 'main' into 162-fix-conda-deploy
ee7fbfc Merge pull request #228 from Living-with-machines/219-annotation-file-paths
53c775d Merge branch '219-annotation-file-paths' of https://github.com/Living-with-machines/MapReader into 219-annotation-file-paths
fbd106b fix test
9bd1595 fix indentation error
7da0034 Merge branch 'main' into 219-annotation-file-paths
0d5daa6 update tests
21cd2c2 Update CITATION.cff
6eee2fe add K Westerling as author
573f9d7 add citation.cff file
2a6e73e error if remove_broken=False and broken paths exist
fec4a0d fix problem of using df_test =0
af6511c fix print full (abs) path for broken_files.txt
bc3a586 us os.path.join to update paths
1160d5f fix pygeos vs shapely warning
ffcae41 print full (abs) path for broken_files.txt
-->

-

## [v1.0.5](https://github.com/Living-with-machines/MapReader/releases/tag/v1.0.5) (2023-07-25)

<!--
PRs: #241, #221, #244, #227, #225, #226, #224, #222

9ed7c86 Merge pull request #241 from Living-with-machines/update_file_saving
e375dc5 Update test_annotations_loader.py
aa69147 Merge branch 'main' into update_file_saving
62f8cba add test_classifier.py update
2d05ddd Merge branch 'main' into 219-annotation-file-paths
445618e Merge pull request #221 from Living-with-machines/dev_train
773b93d Merge branch 'main' into update_file_saving
7384f15 Merge branch 'main' into dev_train
070cd8e Merge pull request #244 from Living-with-machines/rw_docs
cc7a9f3 Update About.rst
e192e77 Merge pull request #227 from Living-with-machines/rwood-97-patch-1
aad59aa Changes conda GH Action to only deploy on tagged commits or when manually triggered
1516e85 Unifies setup.py ".[dev]" install and CI "Install Tools" step
1dc5701 Merge branch 'main' into update_file_saving
7a10511 Merge branch 'main' into dev_train
ceedb77 Merge branch 'main' into 219-annotation-file-paths
1882ace Merge pull request #225 from Living-with-machines/dev_classify
52c0187 fix broken annotate
dc5e466 americanize worked examples
3496baf americanize tests
9e15d5b americanize spelling in mapreader code
274cf16 americanize docs and readme
5be63a0 fix typo
01eecfe Merge branch 'main' of https://github.com/Living-with-machines/MapReader
9ee5e77 Update README.md
4210b7e Update Contribution-guide.rst
b2bb80d add owslib to setup.py
b05cc91 update tests
336e894 allow .tsv files
be8153d fix problem with df_test=0
1adeb8e change all files to comma separated as default
eae3420 update docs
ae7a566 update tests (fix error)
4701271 update tests
e633cab raise error if no annotations remain
00187cf add function to check patch paths
1bee060 update error messages
b9bf1c6 add 'images_dir' argument to load_annotations
04b2153 Update publish-to-test-pypi.yml to only run on review requested
970556f Merge pull request #226 from Living-with-machines/dev_download
ba741c3 make tqdm.auto throughout
fb100c7 add tqdm to sheet downloader
3bcbd83 add test
09c5696 Merge branch 'main' into dev_classify
7e4f032 Merge pull request #224 from Living-with-machines/rw_docs
9892cdd add guidance for timm models
f574956 add dev dependencies (timm and transformers)
46890d5 add tests for inference
37ad30a Merge pull request #222 from Living-with-machines/asmith-paris-prep
83b9635 Updates from download to annonate
8838977 fix errors
4ce652c add tests for hf and timm models
-->

-

## [v1.0.4](https://github.com/Living-with-machines/MapReader/releases/tag/v1.0.4) (2023-06-27)

<!--
PRs: #209, #210

837a842 Merge pull request #209 from Living-with-machines/dev_download
b53672a Merge pull request #210 from Living-with-machines/dev_load
235b5a6 add tests for other models (load from string)
b1d90b3 add saving of state_dict as well as whole model
ade3d71 update error message
052e984 Merge branch 'main' into dev_download
cf85f86 Merge branch 'main' into dev_load
-->

-

## [v1.0.3](https://github.com/Living-with-machines/MapReader/releases/tag/v1.0.3) (2023-06-21)

<!--
PRs: #220

2dcb57b Merge pull request #220 from Living-with-machines/tweak-developer-docs
3415f53 adds details to dev docs about version numbers
-->

-

## [v1.0.2](https://github.com/Living-with-machines/MapReader/releases/tag/v1.0.2) (2023-06-20)

<!--
PRs: #202, #197, #200

055d240 Merge pull request #202 from Living-with-machines/debug-gh-action-event-trigger
aa16de4 add error message for broken image files|
6c4b654 save metadata to csv on each download
28c70d4 enables manual triggers
3199314 Merge pull request #197 from Living-with-machines/general-input-guidance-update-w/NLS-tile-server-update
074b7ad Merge pull request #200 from Living-with-machines/deploy-to-pypi
3ef55bf fixes production PyPI deployment
-->

-

## [v1.0.1](https://github.com/Living-with-machines/MapReader/releases/tag/v1.0.1) (2023-06-06)

<!--
PRs: #188

40d65ca Merge pull request #188 from Living-with-machines/workshop_notebooks
6f24b50 update empty notebook
c100554 changes versioneer style
216da8f update workshop notebook w/ katies comments
820ea4b checkouts full git history for versioneer.py
d70c44a adds missing -m switch
b8ea250 removes poetry as build tool
3853383 takes GH workflows from alto2txt
-->

-

## [v1.0.0](https://github.com/Living-with-machines/MapReader/releases/tag/v1.0.0) (2023-06-06)

<!--
PRs: #195, #199, #185, #182, #164, #181, #176, #154, #180, #163, #160, #165, #151, #155, #140, #95, #158, #74, #149, #79, #136, #133, #102, #96, #94, #91, #87, #90, #89, #86

eb2f0ea Update conf.py (turn off todos)
cedae71 Update README.md
597bcfb Merge pull request #195 from Living-with-machines/rwood-97-patch-1
1fcdc19 Merge branch 'main' into workshop_notebooks
d7ef18a Merge pull request #199 from Living-with-machines/rw_docs
a7e71a9 update worked examples in docs
bd15390 update annotation_tasks file
081e3fa update classification_one_inch
fa52e59 remove coastline example
9502e80 Merge pull request #185 from Living-with-machines/dev_train
e4d384b remove mnist old notebooks
02b4524 update annotations files
cf489a6 update plant pipeline
cc91777 again fix issue with mse loss
b2571dc ensure index has 'image_id' label when saved
5a2dc3f fix error if using mse loss
6cdc6fa move persistant data (NLS) dataset
67604f7 move small_mnist dataset
645e99d update mnist worked example
6386256 Update Input-guidance.rst w/NLS tile server details
b1f94b1 update venv env name from mr_py38 to mapreader
77a8077 fix train_test_split
f4e253b add doc string and inference guidance
3437645 updates from kmcdono2 comments
1cfe86d add annotations
9ce9edf Merge branch 'main' into workshop_notebooks
5bd91ca Merge branch 'main' into dev_train
3907b7d fix to show_parent()
c554fc7 add show_parent() to workbook
6b3d12a Merge branch 'rw_docs'
a406362 update docs for saving geojson and csv files
9f0da34 clear outputs and create 'exercise' notebook
30869d4 update workshop
44de8bc minor fixes after testing
626a928 Update setup.py
457722c fix some typos and add type hints
6c160ee update notebook and create annotations file
84f1574 Merge branch 'dev_train' of https://github.com/Living-with-machines/MapReader into dev_train
c781d66 Merge branch 'main' into workshop_notebooks
94906b8 Merge branch 'main' into dev_train
c6828c8 Update test_loader_load_patches.py
18b4a65 make work with old annotate
26e17bb fix issue with load_patches (ensure patches are added to parents)
61419b1 Merge branch 'dev_train' of https://github.com/Living-with-machines/MapReader into dev_train
6a9e905 formatting
a7c03d9 black
4803d26 Update test_annotations_loader.py
076d219 update pkl files
accf9b3 add tests for classifier
b025d85 add tests and fix deprecation  warning
762bcf5 fix unique_labels if using `append`
49cdc5f Trying to resolve display issue
960ef8d Cleaning notebook
eddbb06 Adding worked example for annotations
a73f157 Create annotations_dir
4173e02 Dropping unnecessary f-string
736e6a2 Adding in a TODO
e09d4c1 Updated `Annotator`
7b04d3b add in-out pics
0e5ee3c minor updates to pics
4693444 fix docs issues in classifier
9107409 finish updating docs
052537f Update README.md
a799cbe fix typos
206884e update docs up to training part
f34c82a Merge pull request #182 from Living-with-machines/dev_load
b45f559 allow use of hugging face (tranformers) models
cfd4d6d Merge branch 'main' into dev_train
725c2a7 update classify name
c239855 Merge branch 'rw_docs' of https://github.com/Living-with-machines/MapReader into rw_docs
d21f3e7 Merge pull request #164 from Living-with-machines/rw_docs
cf685cf Update README.md
6284d65 Update README.md
f0a11fc Merge branch 'main' into rw_docs
4d38640 fix ClassfierContextContainer
0ae91d9 fix PatchContextDataset
86c5a57 rename to classify
48f7610 fix patchify by meters bug
0769916 update workshop notebook and fix bug in images.py
39a8a60 fix patchify by meters bug
43e286a Merge branch 'main' into rw_docs
f9e53b3 initial commit for workshop notebook
7d29956 update tests
2ab8a63 Merge branch 'main' into dev_train
c8208a4 force geopandas to use shapely
fdae2df keep label and pred columns if present when saving to geojson
1fd1842 make shapely version 2.0.0+
2782813 Merge pull request #181 from Living-with-machines/main
53e2f11 Merge branch 'main' into dev_load
891ad79 add tests
8a5f343 Merge pull request #176 from Living-with-machines/174-save-metadata
f71f193 add torchinfo to requirements
321242b update doc strings
878401c fix loading preexisting model
c9e1f0b fix initialise model
c55a05a add labels map to annotationsloader
add408c allow no label col/label indices in patchdataset
fed479e fix classifier inference
0af9443 Merge pull request #154 from Living-with-machines/dev_download
7f91c60 update classifier
87f8ed8 add label indices to patch dataset
b0f7819 update classifier class to ClassifierContainer
bf28c3f fix assignment of datasets variable
aab1fe5 add todo comment in train docs
5fcf6c5 set up samplers by default in 'create_dataloader'
a3653af Name set to next/next random patch depending on settings
2b5828c Adding margin as keyword arg to `annotate`
da222e6 add 'create_datasets' and 'create_dataloaders' methods to 'AnnotationsLoader'
b7d18ff fix PatchDataset in datasets.py
b347db4 fix tests
74dec99 fix import tests
f01284f create new branch (mock_downloader) for mock downloader and rm file from this branch
f0598c7 update docstrings and ensure PatchDataset methods work
c010ff1 save patches as geojson
dc9ff64 Merge branch '174-save-metadata' into dev_load
11a5aed Merge branch 'main' into dev_load
a200d5b save to geojson
06ac58d Fixing detail
b38a03f Adding auto-resizing of patch images to 100px
d1ae0e8 Renaming `"changed"` column `"annotated"`
3a24e91 Adding TODO
0bb1b9d Fixing annoying spelling error
2cf4d12 Fix typo again
4e6c0c3 Spelling fix
cd0066c Adding in a TODO
536b249 Renaming frames + ensuring annotation_dir exists
1bb9dfc Fix bug
4aeeabc Adding debug messaging
bc1cddc Adding `metadata_delimiter` keyword argument
56118e4 load_annotations now returns PatchDatasets
7f7a5dc Merge pull request #180 from Living-with-machines/dev_load
89ca44c rename 'kwds' to 'kwargs' to follow convention
46972b1 move load_annotations to learn
e7bf85e minor fix to test_sheet_downloader_mock_example.py
67c56e8 Create test_sheet_downloader_mock_example.py
77abf4c Add tqdm to calc_pixel_stats
e840736 add tests for saving dataframes
8a8a967 fix crs key name
f12740b fix tests
d960dd8 add tests for crss
3e6e69b update query_by_string tests
f042fbc fix grid_bb to/from polygon (remove self as arguments)
6abc191 add comment re. temp_folder
2aa0216 set temp folder in tile_loading + import elsewhere
92be697 update error message
fe4a225 fix ordering
6a90868 add grid_bb to/from polygon to downloader_utils
626b74f update docstrings in dowloader_utils
48ed7f6 fix sample files
7e58296 Refactoring as queue + UI design
550bf82 ensure CRS in patch_df
ea4eaec Merge branch 'rw_docs' of https://github.com/Living-with-machines/MapReader into rw_docs
843e490 fix typo
7dfc8b6 update env name (goodbye mr_py)
7142227 ensure dictionary keys are same for crs
edeec51 raise error if your coordinates are not 4326
b4f2e70 fix tests
e2fa907 ensure CRS is retained in metadata.csv
77c824e update download.rst (user guide)
ab04ccc update input guidance
c688a76 fix xxx_by_string() - allow no keys/keys as string
05bd81d make csv default save as tab separated
0a3888c Update Beginners-info.rst
e73d906 Update README.md
1aca675 Update Install.rst
db10d04 Update Install.rst
cd5a8b2 Update README.md
8ed7f11 Changing `min_values` + `max_values` to `mean_pixel_RGB` in example for `annotator` method
497c1e2 Filling out example better
8202e65 Adding examples to `annotate` method
dc29624 Changing UI a bit further
347d04e Bugfix
7ae518c Refactoring, more UI, better filtering
f5a481b Refactoring the code
084a7fe Changing look of progress bar
4132780 Adding showing of `url` (and fixing some bugs)
dccbc9a Adding ability to filter (like `min_mean_pixel` and `max_mean_pixel`)
ce4ab75 Adding docstring for `sortby`
33cd926 Adding `sortby` keyword argument
f811cea Keep patch filepaths + keep label names in output
cf21d61 update docs requirements and conf
963b916 Merge branch 'main' into dev_download
72ede17 Merge branch 'rw_docs' of https://github.com/Living-with-machines/MapReader into rw_docs
04a4f54 fix tests
9000bf4 add 'save' argument to convert_images method
3035caa fix typo
d6b6cce More docs additions
40f6454 Adding in a few notes in the docs
30efdd1 Update conf.py
d2b4b04 moving `Annotator` from `annotate.utils` to `annotate`
477fff0 Adding a progress bar
b9e79a0 Adding `show_context` option
c136f3d minor updates/ typo fixes:
1042d70 add todos
6282329 ensure contribution guides included in sidebar
cb8bba3 update load title
a223298 update load with advanced usage and to reflect updated code
c8b58e1 update extensions
3e00554 Merge branch 'main' into rw_docs
c8ad17f Spelling mistake in typing
81092a1 Merge pull request #163 from Living-with-machines/dev_load
4014304 Clarifying `annotations_file` attribute
b2933c3 Merge branch 'main' into dev_load
d8c3b77 Adding a missing parameter (`stop_at_last_example`)
6ba561f Merge branch 'main' into dev_load
0895d35 Adding in some typing
cf9a209 Dropping unnecessary and conflicting import
8c84cb4 Formatting
06db6a4 Adding in metadata + fixing docstring + little bug
fe5b15b Merge pull request #160 from Living-with-machines/131_geotiff_bug
4e55d65 Fixing a tiny bug
06f87da Adding an example
9183603 First commit of new `Annotator` class
3c47fa7 update tests and init
003001c remove download_old
5b8dba5 rename download/download2
88c7200 merge test_images and test_images_add_metadata
2922deb rename proj2convert to target_crs
c251531 add tmp_paths
377871e add tests for PIL.UnidentifiedImageError
b50bd5b add specific exception handling
3e66d80 rename test_dirs to dirs
505605c make images_constructor private
2c0cc6f update reproject_geo_info
7547c2e add note about overwriting existing metadata info
1b3a99f rename tests
5a061ec update file names in tests
88ff09d rename sample tif files
a5e7455 update saving of coordinates for metadata
0c5e3f2 fix tests
1d77fd4 fix typo
4af03e3 update downloaders
f701870 final changes - include CRS as attribute
2a89615 save patches as ".tif" only (not ".png.tif")
901c6a8 Merge pull request #165 from Living-with-machines/add-code-of-conduct-1
dac6446 save patches as geotiffs
5b92f5c add beginners info links
5b44aab Update test_load.py
e1a0f72 add beginners info text
53552b3 fix tmp_path not defined
bb61c97 add error if you try to load non-image files
7142155 update tests with excel files
2581a3b allow metadata as excel file
55b1939 add beginners info + update input guidance
4abf020 add tqdm to setup.py
98bd79e minor fixes
037d093 Add print info for where patches are saved
44b49ea add tqdm and update show
5d7fe95 update toc trees - local only
73585d8 update toc trees
6b611db Update contribution guide into tutorials/docs/code
fe08a7b Update test_load_loader_add_metadata.py
348676a add developers guide to docs
dde3171 Merge branch 'main' into rw_docs
3654ff3 fix add metadata tests (checkout file from ``dev_load``)
f39157a Merge branch 'main' into 116_data_inconsistencies
b99f39e Merge branch 'main' into 131_geotiff_bug
8e50742 Merge pull request #151 from Living-with-machines/116_data_inconsistencies
e9b8298 Merge branch 'main' into 116_data_inconsistencies
18bfa58 move geo_utils to load subpackage and fix tests
39d08b7 Merge branch 'main' into 116_data_inconsistencies
41a168d fix tests
73374b4 fix rename of ``update`` to ``overwrite``
cfe2c4e Merge branch '131_geotiff_bug' into dev_load
5232ada update load_xx methods
dd8c1fe reorder to help with comparision to main
4982070 Merge branch '116_data_inconsistencies' into dev_load
42d4867 fix 'not a PNG file' error
5e7b51d add tests
85d4b4d Merge branch 'main' into 131_geotiff_bug
32cfa19 update tests
46bd32d Merge pull request #155 from Living-with-machines/adding-kallewesterling-to-project
1ee4087 Merge pull request #140 from Living-with-machines/110_loader_file_paths
6d1273b Merge branch '110_loader_file_paths' into dev_load
7df502f Merge branch 'main' into dev_load
562a66a fix tests
5ba7dcb update tests for sheet_downloader
e8769f6 rename test
84d8c85 fix assertions where keys are different
4c935bf update tests and add option to ignore_mismatch in add_metadata
dfe9352 woops sorry - rename tests
0097a6a rename tests
68861ad rename test files
a5a2bfb Merge branch 'main' into 110_loader_file_paths
4167524 update tests
fbcb17c make resolving paths a separate static method
3c3d24a updated test_loader_filepaths.py
383938b Merge pull request #95 from Living-with-machines/82_align_task_names
9c385b9 run black
f528266 update init + imports
aa46c4d auto pull in extra info when constructing images
19dde7c add patch coordinates
ab0a2b5 method for verbose printing
0ad8e9e method to get tree_level from image_id
f9f98a9 rename/edit `add_par` to _add_patch_to_parent
27287e9 edits to add_xx and calc_xxx
128f4b5 update slicing methods (slicer now obsolete)
976f3c2 move geoinfo method up + split for individual imgs
a966185 edits to show/plotting methods
4ff9f99 create parents and patches attributes
bae9849 Update images.py
8d15b22 update calc_pixel_stats
34db6b3 update calc_pixel_width_height
3743520 update images_constructor
8e91236 update image_constructor
9d83ef6 Merge branch '116_data_inconsistencies' into dev_load
db9bbf6 Merge branch '110_loader_file_paths' into dev_load
78df517 align to PEP8 style guide
6919a8f Merge branch '82_align_task_names' into dev_load
8533b85 minor updaes to images.py
dd4212c ensure coordinates are xmin, ymin, xmax, ymax
06a4b90 fix coordinates assertion error
00311f4 update tests
79abfd8 fix so coords are actually xmin, ymin, xmax, ymax
0afca61 Update publish-to-conda-forge.yml
08bff60 Merge branch 'main' into 131_geotiff_bug
096dacc Merge branch 'main' into 82_align_task_names
36524ff Merge branch 'main' into 116_data_inconsistencies
5945678 Merge branch 'main' into 110_loader_file_paths
cc65278 keep private label for '_xxx_id' methods
b678b5d Merge branch 'main' into 82_align_task_names
5d44cf5 Merge branch 'main' into 116_data_inconsistencies
b0f0157 Merge branch 'main' into 110_loader_file_paths
5fb19c7 Update publish-to-conda-forge.yml
5c090d3 add error message for not implemented image modes
cf308c3 Merge pull request #158 from Living-with-machines/141_id_methods
6a608b6 set show progress = false for all
8dac08b add explicit timeout to fix windows test failure (hopefully?)
42d9b27 missed one
0605849 make xx_id methods private (i.e. _xx_id)
40fca50 add utils to init
18d6ca1 simplify download methods
6becbbd Update ways_of_working.md
fcc1369 add query/download by string
c4432a2 fix "set.isdisjoint" again
c446b8b fix 'disjoint'
0570af4 fix duplicte query results
02c2615 add tests for query/download by line
3101a7e add print arg. to queries
3bd6ff5 add query/download by line
e6fe9be Fix plotting (again)
d1ffa3c fix add_id
a82c30d fix cartopy/plotting issue
b8d6414 add cartopy to setup.py
3709dca Update publish-to-conda-forge.yml
5db987a add tqdm to setup.py
5fcf9e5 add tests
5b0759b Merge branch 'main' into dev_download
06edfdb add to __init__.py
6a4dc82 minor fix to downloader
b675ad7 update downloader
d75a817 add error/warning messages
48a13b5 doc strings + only download if not already exists
89a07d6 black
8032feb allow download via queries list
af78537 Add option to query maps
e167c5d black
aa04a72 add writing of metadata file
95ab010 fix downloader (no metadata)
920a004 save filename as map_name
9e2a3b6 add download2 - alternative download option
cea7043 Merge pull request #74 from Living-with-machines/feature-conda-package
3ea7912 fix error if index_col is image_id_col
0b36eb0 add tests
afca1a5 update images.py with warnings for data inconsistencies
aabc668 update docs
a4983ba fix typo (again)
3bb9c69 fix typo
2336322 Merge branch 'main' into 82_align_task_names
0b30db4 Merge branch 'main' into 110_loader_file_paths
a1524d8 add api docs to gitignore and rm files
9bde40d Merge branch 'main' into 82_align_task_names
71d377e gitignore docs/source/api/mapreader and rm files
2bd1651 remove api docs
6dd921f update gitignore
bba5822 Update README.md
542eca2 Merge branch 'main' into feature-conda-package
1d1394c updates to docs
c03dccd update tests + fix errors
18f1e9c update code after merge
a4adfce Merge branch 'main' into 82_align_task_names
451230a Merge pull request #149 from Living-with-machines/fix_docs
fbe81b3 fix formatting of scraper and stitcher main text
996f314 rebuild
80a10c9 change type-hints to description
0806b6b minor changes to .py files
928d650 fix bullet list in tileserver_scraper
faa7bd2 fix code-block warning
a74609c fix invalid imports in utils subpackage
e753eb5 remove autosectionlabel from conf.py
8cec91a move datasets to worked examples
5ba8120 explicitly set conda env in every cmd
101b99a Attempt to fix python version for `conda-build` in GH Actions
546d15b add tests for importing various geospatial modules
35bd710 source docs version from package version
c2c4a3e Fix version number resolution
c0a08fc Comments/updates from meeting with Jon
fecf874 Minor changes to docs
c0a23bc Merge branch 'main' into feature-conda-package
cff636c Add geo dependancies into conda
048495a Merge pull request #79 from Living-with-machines/76_enable_geotiff
f1420f4 run black
7712678 add tests 4 loadPatches, loadParents +load_patches
6744b86 Add file extension arguments to 'load_patches'
206d312 Add tests for loader()
65442c0 Update setup.py
1d614bc remove examples
69ec22b Merge branch 'main' into 76_enable_geotiff
a2d312c run black
74fe2d1 add for loadPatches and loadParents, error if len(files)==0
9bb4fc8 allow no file_ext for dir. with only one file type
e19a4c0 add rules if directory passed as path_to_directory
81aa686 fixed typos
0d56fa2 train to learn
09a116b fix file_not_found error
f93243a update default path_save for patches
0e0e929 updates to docs
9d453b3 update docs
8643408 pull updated setup.py from main
473c593 Merge pull request #136 from Living-with-machines/remove_geo_install_option
d55a916 Merge branch 'main' into 82_align_task_names
adf061d Update authors in setup.py
04511ef Update setup.py
53e60bc update setup.py
6ad3c17 update colorbar
751aec0 Merge pull request #133 from Living-with-machines/rw_docs
0f643ff update figures with all one inch examples
3d1572b add examples to train and update explanations
a9a0ef4 update docs (mostly train)
762ba94 updating user guide with directory structres and #EXAMPLE labels
77d123d fix formatting in contribution guide
0038c5f updates to about
d70b731 more updates to about and input guidance
73ceeb1 fix install
45198ba Update input guidance with metadata info
dc5e565 updates to about page
02d3ae6 explain what is pipeline and add fig
95a1794 update install instructions
908708b update contribution guide +requirements for sphinx
6e822a4 fix sphinx explicit target warning
6d8c92b minor updates to formatting
6ef22f2 run blacken-docs on all rst files
f15de74 Adding documentation to `mapreader.loader.*`
6c87532 updates to toc-trees
027dcdb Update annotate.rst
ede7afb update load.rst and fix error in user-guide.rst
2ce755f Fixing a little linting.
5540a50 Adding tileserver_access documentation
a70dd3a updated input_guidance
3bb365a Editing documentation for consistency
9bba1db Fixing `pytest` error for `typing.List`
06e14fa Adding a note about matplotlib for the `plot_sample` method
e870b02 Documentation added for `train` submodule
d621223 Cleaning up latest commit
70f32fd Documentation added for `download` directory
d325c37 Documentation added to `process` folder
5c7cbd3 Documentation on the `annotate` folder
ad4867f Documentation added to `annotate`
b65a0c3 polygone to polygon
632b3a8 Update Contribution-guide.rst
a60e6d0 Update documentation_update.md
5af13ea Create documentation_update.md
87f21b3 update annotate.rst link to paper
4485f4e Rename Load_Patchify.rst to Load.rst
45222d0 Update input guidance - fix headings
28e104c Merge pull request #102 from Living-with-machines/rw_docs
d4a7b39 update annotate.rst with KM's comment again
dd51004 fix indentation error in annotate.rst, move examples explanation to user_guide.rst
2c4b2aa fix link again agin
78f24a4 fix link again
66eb457 update paper link
e1efbb5 tell people where a template of the yaml file is
3f26ced some text about patchifying + comment
9b46c15 just some small style changes and an apostrophe :)
a4f1084 Update Download.rst
6a55888 small update re: cloud services
da3a08f made some new comments
2acdeb8 updated formatting for comment at end
b26dbee update filepaths in test_non_geo
b51896f update download to say we are using six inch
2ec5281 depth==1 of toctree in worked examples
a8879b8 add worked examples notebooks to docs
44ace29 Merge remote-tracking branch 'origin/98_worked_examples' into rw_docs
9ce351b Merge remote-tracking branch 'origin/main' into rw_docs
1907fba remove quick start
079c04b update all notebooks to align with docs
1cca320 Update Input-guidance.rst
f5e9171 adding todos
d84e3aa updated notebooks in classification_one_inch_001
30722d2 Update ways_of_working.md
9739766 fix table of contents
d2becf7 added old text from geospatial README
ba83da6 updates to train images (with updated transformations
2c840f2 add comments re. notebooks and filepaths to annotate.rst, update and fill in gaps in train.rst
d544abd add notes in load.rst re. using notebook and updating filepaths
6147ad3 update download.rst to clarify commands are for python IDE and add clarity re. querying
d59cc40 update install instructions to be more explicit
4d4720f add coastline worked examples - download and load
2a7b987 move persistant data, rename examples to worked_examples
6aee20e remove docs/source/api/ from .gitignore and readd files
bd21eaa update api docs to reflect changes
3cd1f47 Merge remote-tracking branch 'origin/main' into 82_align_task_names
563ebb5 re-add api docs
84b5e0d load patchifyByPixel inn image.py
bb98411 Update .gitignore
b3c353a update docs and tests to reflect with name change
e544b07 move slicer.py to load subpackage and rename slice functions to patch/patchify
10fbcd0 Merge remote-tracking branch 'origin/main' into 82_align_task_names
fc75548 run black on all files in mapreader/ and tests/
ba923b3 Update README.md
6f53802 Add files via upload
d3bf275 Merge pull request #96 from Living-with-machines/rw_docs
ceb01be add docs/source/api/ to .gitignore + clean current files
faec940 add brief definition of the word 'patchify' to load.rst
eefe18a rename utils.py to geo_utils.py, write unittests for geo_utils functions, ensure imports reflect name changes
c560abc Update ways_of_working.md
5d39817 Update ways_of_working.md
cd41dd0 Update ways_of_working.md
bd3f81c Update ways_of_working.md
235cd17 Update ways_of_working.md
879c2ed comment about path for metadata
8942f94 comments about doc
a67dbf8 update geoinfo functions in loader.py and split extractGeoInfo function into two in utils.py
239304b update test_loader.py to include pytest.fixtures, increase approx tolerance on coordinates, replace abs file paths with relative
1f8b92f fix indentation error in  (images.py)
f41cf3d update images.py - remove  option from coord_increments methods, always print warning messages on  and , simplify
eba26ea updates to error messages in coordinate functions
08eae76 added docs/build/ and docs/source/api/ to .gitignore and removed files in these directories
a4b1884 update doc strings to ensure clarity
fadbb46 rename h,w,c to image_height, image_width and image_channels and ignore unused variables when unpacking shape (h,w,c) tuple
39d9e99 run black on all files in ./mapreader
e118800 rename child/children to patch/patches
b3b7df1 rename loader to load
93a33ec Merge pull request #94 from Living-with-machines/71_fix_print_dataset_name
5fc0925 ensure 'set_name' is specified when calling my_classifier.show_sample()
ccda8ec remove docs/build/ and add to .gitignore
980d617 Merge pull request #91 from Living-with-machines/rw_docs
2e554c0 fix setup.py
3806998 delete pyproject.toml
740fcb4 updated my details
4f7debf fix links for bug report and feature request
9bf4fdd v. minor tidy-up of conda specification
09be1f4 Remove accidentally committed env files
412bf1c Update README.md from main
db402b1 Merge remote-tracking branch 'origin/main' into feature-conda-package
72ccb46 Update docs for conda install method.
0fd77ed update User-guide, create contribution guide doc and include in index
ebdca62 Update README.md
fea54d6 Merge pull request #87 from Living-with-machines/81_update_README
d3ab11f Update ways_of_working.md
7e1ba22 Merge pull request #90 from Living-with-machines/80-ways-of-working-asmith
11690fd Merge pull request #89 from Living-with-machines/88_fix_npfloat_warning
0bc2923 added details to ways of working doc
c9f78fd update annotate docs to more generic use cases + separate worked example for rail_space
93af00c change np.float to float
27cfed1 updates to User-guide docs
6dda45c updates to User-guide docs + index.rst (now pulls from README)
4e9d93b center align Fig 2 caption
9d5c5fb center align Fig 2 caption
d356506 Merge remote-tracking branch 'origin/main' into rw_docs
5dfa98e update docs to align with updated code
74c6446 remove testing of show() and show_par() functions to stop tests hanging on windows
d71ffbc add pytest approx to allow for rounding errors
2c8f3de add tests for loader subpackage
555822d Merge pull request #86 from Living-with-machines/80_ways_of_working
844f26d update with kasra's details + acknowledgements
e8c2c21 add conda env exports for different configs
704a961 updated README, gallery removed (will move to readthedocs), new picture tbc
ef2080d Run bash as login shell (to enable conda activate cmd)
05d182e create custom conda env for build
756e0b3 Add --skip-existing for main upload
1118f17 Add extra import check
a32bd5e Convert to multiple platforms and upload
8890213 Update ways_of_working.md
d5f48d4 fix paths
ba361ca Move mapreader meta.yaml
76eb4d4 Add --skip-existing to upload cmd
9a576f1 Attempt to build mapreader using pre-uploaded dependencies
37efe3d Update publish-to-conda-forge.yml
16129c5 Install anaconda client
815745a fix typo in cph transmute command
712778f Enable publish dependences to conda
12d3334 Tidied up comments in meta.yaml
e82cd26 Fetch all history in publish-to-conda-firge.yml
e3598a0 add conda-forge as source for conda build cmd
9a877ad update annotation files to reflect code changes
dbbcda3 Fix error in commands for post build tests
65c33e2 Add GH workflow and post-build imports for tests
3d2a2cf update save format to png
bf2c46a restore examples from fddca82
57b5b08 tweak to version tagging prefix
f71620b add build details to developer's docs
19cbae7 add ways_of_working.md
-->

-

## [v0.3.4](https://github.com/Living-with-machines/MapReader/releases/tag/v0.3.4) (2023-02-20)

<!--
PRs: #69, #68, #65

e1355d3 Successful local build of conda package
89ea886 build conda packages for upstream dependancies
477e048 cherry pick a14bec1 (Add min_std_pixel and max_std_pixel)
e9ecb55 Create PULL_REQUEST_TEMPLATE.md
56c38d4 Update issue templates
7aec00a comments on README vs docs homepage
7b0094e remove cartopy from pyproject.toml
9078a9f updated in line with updated loader function
547bcba tweak conda build tests
71dd178 added rasterio to dependancies to stop tests failing
bfa778e added options for verbose
1e62b70 updated loader with method to add Geographic info from Geotiffs
d8ee850 FIx proj and pytorch conda dependancy resolution
bb1b4cd updated Load.rst with option to add metadata from df
dfe2f1f Add versioneer
9526351 Add issue templates
a9e14fb updates to install instructions after removing cartopy from dependancies
10e8b76 updates to install instructions
471ab9d updates to install instructions
166923e Update Input-guidance.rst
362ab99 Merge pull request #69 from Living-with-machines/rw_docs
dacfe5d updates to train.rst after testing
7c895f8 Merge branch 'docs_train' into rw_docs
635dbc4 updates to docs after testing
f875a07 update Train.rst after going through plant tutorial
0b0f45f Merge pull request #68 from Living-with-machines/docs_load
7f7ade4 updated Load.rst after going through plant tutorial
86f9593 Merge pull request #65 from Living-with-machines/rw_docs
740731a added basic docs to Train.rst using tutorial as base
fd1bfac added copybutton + built html
0cc29ce added copybutton
20c2f28 removed process/patchify and testing 'sphinx-disqus' extension
828f503 updates to train.rst
fb9a155 first thoughts
6736576 organising docs files into separate directories per 'section'
3812dd5 added basic docs to Annotate.rst using tutorial as base
f3b564e fix images in Load.rst
e5c77b2 added basic docs to Load.rst using tutorial as base
a87f5cb added basic docs to Download.rst using tutorial as base
8250548 Create how_to_contribute_to_docs.md
c11767e adding download section
69e5c7d Update Load.rst
089a50e Update Load.rst
3d92da8 setting up docs structure
55dec91 fix requirements.txt
8f6c1ea added READMEs
badf33b update index.rst to include READMEs
2748025 adding README.md
c693641 add requirements.txt
736eaec adding .readthedocs.yaml
059bf9c initial commit
52ea967 initial commit
c0a08ff initial commit
fddca82 Update README.md
c779d4f change links and citation from arxiv to ACM paper
c5245a1 Create CODE_OF_CONDUCT.md
f17d283 Add CI for pip install
d8f079d version 0.3.3
-->

-

## [v0.3.3](https://github.com/Living-with-machines/MapReader/releases/tag/v0.3.3) (2022-04-27)

<!--
PRs: #38, #41, #40

395a41b 5 epochs
5116710 add annotations
3abcec8 update README
9e2fabf Update README.md
0545145 v0.3.2
8eb88be v0.3.1
529a121 Merge branch 'main' of https://github.com/Living-with-machines/MapReader_public into main
2a5ffd3 Update annotations for classification_one_inch_maps_001
a7964a3 Update README.md
eefb584 Update README.md
8db49ba Update README.md
4b53996 Add style
063155d review notebooks
651a2d1 change maps to geo in installation
6fbc912 Update README.md
eda5322 Update README.md
01537ad Update README.md
a2394cd Update _config.yml
118ec2f Set theme jekyll-theme-minimal
b68a4fc Set theme jekyll-theme-cayman
0ef28b1 Update README.md
ec58490 Update README.md
efab91b Update README.md
c32bf45 Update README.md
8411085 Update README.md
d4ecfeb Update _config.yml
2c2ec90 Update README
f87bda6 Merge branch 'main' of https://github.com/Living-with-machines/MapReader_public into main
03cdcd1 Update README
0cb64ef Update README.md
75aded4 Update README.md
b4d649c Update README
d086fcf Update README
06c0bd1 Update README.md
500129b Update README.md
c47a088 Update README.md
9c61e1d Update README.md
02fbaa7 Update README.md
8220bcb Update LICENSE
ec4f290 Set theme jekyll-theme-leap-day
7c268d1 Merge pull request #38 from Living-with-machines/restructure-readme-kh
d9f8178 Update README.md
6bee71d Update README
2036e86 Update README
b80b93e Update README
3fd92b5 Update README
932d27c Add JVC paper
faf1357 Update README.md
f690f75 Move TOC after gallery
a7eb4b6 README: geographic information science
115dcc4 Add windows-latest to CI
85a6865 update README
2d0d3c2 update README
1e462b3 update README
dfbd365 update README
8670fd5 Version to 0.3.0
3ed973e Remove as it is empty
5da2e49 Add .[maps] to CI
86dab42 update CI
b3bb835 update CI
b67f01b black
a670c33 black
95f2ab8 Update README.md
6a058ca Add MapReader paper
bce00b1 Add a new test
3d4a77e Remove README from classification_one_inch_maps_001
700a6fb Update README
5dad9fb Move README to geospatial
8dfe471 Merge branch 'restructure-readme-kh' of https://github.com/Living-with-machines/MapReader_public into restructure-readme-kh
bff7933 update README
bf670a5 Create README.md
2d804a9 Change map/non-map to geospatial/non-geospatial
d172548 Change map/non-map to geospatial/non-geospatial
bfd0f0b minor
86eb6f6 Merge pull request #41 from dcsw2/patch-2
5217402 Merge branch 'restructure-readme-kh' into patch-2
752c325 Merge pull request #40 from dcsw2/patch-1
087c9aa d actual edits to first para
74481e1 d test making apr
b9b428d Update README.md
62b3d84 Update README.md
4706665 BUGFIX: annotation interface when working with images (and not patches)
135173d Update README.md
e596539 Update README.md
c05960b Update README.md
eeaf086 Update README.md
bbd1717 Update README.md
1f542a5 Create README.md
2c7817c Update README.md
fb0708c Update README.md
1188796 Create README.md
f07ff6b Update README
aae8ff1 Update README
189360f Update README
27c96b2 Move plant notebooks to non-maps dir
f3489d9 Add figure for MNIST tutorial
e43e1fe Add MNIST example
c8329f5 Move maps notebooks to examples/maps
b062b6f Change version to 0.2.0
4d19b4c Update README
b3e35ae Update README.md
0f3f5e3 Update README.md
624f5a0 :bug: fixing empty notebook
2a90331 :monocle_face: adding notebook to branch
a3cb498 Update README.md
53c90a4 Update README.md
b1be3af Merge branch 'main' of https://github.com/Living-with-machines/MapReader_public into main
da0dad0 Move rasterio to map dependencies
4de4403 Update setup.py
9b0c1a2 Move map-related libraries to extras_require
1e0f4f9 Update README.md
-->

-

## [v0.1.2](https://github.com/Living-with-machines/MapReader/releases/tag/v0.1.2) (2022-03-03)

<!--
PRs: #25, #24

41f924a Add files to quickstart notebook
050fe7e Add quick_start notebook
582f17e Update setup.py
54ec4d2 Update setup.py
af72d96 udpate CI
6c5582e Update README
9cb3ed8 Add setup.py, remove .lock and .toml files
29d2a04 Merge pull request #25 from Living-with-machines/iss19-model-inference
a1befc0 Add max_mean_pixel to the annotation interface
a8e0f51 Merge branch 'dev' into iss19-model-inference
3546238 :package: adding  and  file to branch
f317a13 :package: adding  to
259c949 :pushpin: downgrading usage: pyproj [-h] [-v] {sync} ...
9212d5f :pushpin: pinning usage: pyproj [-h] [-v] {sync} ...
10e288c :package: removing  package for binderhub
240d39c :package: adding   package for binderhub
07e8b31 Merge branch 'dev' of https://github.com/Living-with-machines/MapReader into adjust-poetry-installation-instructions
3e3dbf2 :package: adding requirements.txt file without hashes for binderhub build
1b5ee4e Add parhugin v0.0.3 to .toml file
4ca1adb Merge pull request #24 from Living-with-machines/iss-torchvision-import
5c25131 import torchvision needed for show_sample method
4d9842a :twisted_rightwards_arrows: merging branches
d533dda merge with main
9efff5a update requirements.txt
634eed1 update requirements.txt
a9aed7d update requirements.txt
70d7d54 update poetry files
8c16895 add requirements file
9d4a88a Update README.md
-->

-

## [v0.1.1](https://github.com/Living-with-machines/MapReader/releases/tag/v0.1.1) (2022-01-15)

<!--
PRs: #14, #12, #11, #8, #6, #3

17078fe Update README.md
2046504 Merge pull request #14 from Living-with-machines/evangeline-corcoran-patch-1
8d772ea update toml file
dd485ba update fig for README
775d0dd add fig: mapreader_paper for README
4587aef Update README
169b1e9 Update README
7940e3f Update README
39c83fb Update README
5db105f Merge branch 'evangeline-corcoran-patch-1' of https://github.com/Living-with-machines/MapReader_public into evangeline-corcoran-patch-1
2ac07fe Update README
c3c90e1 Update README.md
d750a59 minor
7ffb1dc Add figs for gallery
9ea1780 Add some annotations
1a44344 Update notebooks 3 and 4 to be similar to the plant phenotyping examples
b2c3ccb Clear cells' outputs
81f84a6 Add plant phenotyping example to README
e63b99d Update poetry.lock
42f52e4 Update README.md
5481e45 Add plant phenotyping example notebooks and data
32f0733 Update README.md
2c5d7a6 Merge pull request #12 from Living-with-machines/enhancement-add_min_mean_pixel
04d476d Add min_mean_pixel argument
113162e Update README
809f362 Update README
1fc2ab0 Merge pull request #11 from Living-with-machines/enhancement-check_README
b336174 Update README
8913715 update poetry files
c6425f7 Merge pull request #8 from Living-with-machines/kasra-hosseini-patch-2
96730e8 Merge remote-tracking branch 'origin/kasra-hosseini-patch-1' into adjust-poetry-installation-instructions
5321dc0 :memo: adding additional kernel installation instructions.
e9b6536 Update README.md
cdce5c0 Update README.md
70a16c5 Update README.md
6eb39bf Update README.md
f487dab Update README.md
ca2909f Update README.md
a84c650 Merge pull request #6 from Living-with-machines/kasra-hosseini-patch-1
ca707ee Update README.md
e1b802b Update README.md
f8f8ffe Set theme jekyll-theme-minimal
6185b5d Update README.md
cb80247 Merge pull request #3 from Living-with-machines/dev
aeaeaa6 adding additional Poetry version 1.1.11
bb59383 Add a notebook to compute the density of railspace patches
ecafc06 Add metadata for OS Six inch
03a04a4 add a notebook for model inference
736317b update lock
41a4f13 Add train/fine-tune notbeook
9beeaa3 Add metadata for OS 1-inch second edition
ddf084c add tensorboard
a6b602d Add an example annotation file
44188cb Annotation notebook
abe9a77 Minor changes
a4e6e1c Minor changes
2b482ef Define try_cond1, try_cond2
9615dfd First notebook to retrieve, patchify and plot maps
96720c2 add a yaml file to define annotation tasks
5a0877c update toml
6360272 Update README
eebd570 Update README
c053486 Update README, new re-use license
112091d Update README, new re-use license
890fd14 Update README, new re-use license
db5feae Update README, new re-use license
261a578 add tileserver_access.py
7262ab9 add images.py
f79b396 Update re-use links
8332a1d Add re-use terms for metadata file
1d26db9 update README
cdde352 update poetry files
df5f7be add classifier.py
5e7369f add datasets.py and classifier_context.py
a85522b add utils
490badb add MNIST data
9a04593 add init
ee99178 add process.py
3039750 add __init__.py
770fcc9 add loader.py
5e182ba add slicers.py
442ffef add custom_models.py
320da82 add azure_access.py
468246d add tileserver_helpers.py
18c0e20 add tileserver_scraper.py
9eaff39 add tileserver_stitcher.py
8e5449c add utils.py
7066dc8 add load_annotate.py
4b97933 add init
f34f02b update README
13056b1 update README
01c6fd4 update CI
3250e9d Add peotry lock and toml files
2e1673b Add MapReader pipeline fig
6788c22 README
2fa82b4 add CI
fbdaabf add test_import
e7ee189 add __init__
b37d1dc add .gitignore
1921585 add LICENSE
-->
-
