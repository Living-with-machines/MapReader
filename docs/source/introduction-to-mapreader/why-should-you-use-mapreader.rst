Why should you use MapReader?
=============================

MapReader becomes useful when the number of maps you wish to analyze exceeds the number which you (or your team) are willing to/capable of annotating manually.

This exact number will vary depending on:

- the size of your maps,
- the features you want to find,
- the skills you (or your team) have,
- the amount of time at your disposal.

MapReader uses computer vision (CV) models to extract information (class labels and text) from map images.
This enables users to generate datasets for large corpora of maps in a fraction of the time it would take to annotate them manually.

If georeferencing information is available for the map images, MapReader can create georeferenced outputs that can be linked and analyzed in relation to other geospatial datasets (e.g. census, gazetteers, toponyms in text corpora).
This allows users a new way to explore and analyze their map collections.

Understanding the limitations of MapReader
------------------------------------------

Deciding to use MapReader means weighing the pros and cons of working with data that has been inferred by a computer vision model.

This inferred data can be evaluated against expert-annotated data (i.e. ground truth data) to understand its general quality, but users should be aware that in the full dataset there *will necessarily be* some percentage of error.
As such, MapReader may not be suitable for users who require completely accurate data.
