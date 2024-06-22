---
title: 'MapReader: Open software for the visual analysis of maps'
tags:
  - Python
  - image classification
  - computer vision
  - deep learning
  - computational humanities
  - digital humanities
  - maps
  - history
authors:
  - name: Rosie Wood
    orcid: 0000-0003-1623-1949
    affiliation: 1
  - name: Kasra Hosseini
    orcid: 0000-0003-4396-6019
    affiliation: 4
  - name: Kalle Westerling
    orcid: 0000-0002-2014-332X
    affiliation: 1
  - name: Andrew Smith
    orcid: 0000-0002-4465-2284
    affiliation: 1
  - name: Kaspar Beelen
    orcid: 0000-0001-7331-1174
    affiliation: "1, 3"
  - name: Daniel C.S. Wilson
    orcid: 0000-0001-6886-775X
    affiliation: 1
  - name: Katherine McDonough
    orcid: 0000-0001-7506-1025
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
    corresponding: true # (This is how to denote the corresponding author)
affiliations:
 - name: The Alan Turing Institute, London, United Kingdom
   index: 1
 - name: Lancaster University, Lancaster, United Kingdom
   index: 2
 - name: School of Advanced Study, University of London, London, United Kingdom
   index: 3
 - name: Zalando SE, Berlin, Germany
   index: 4
date: 14 December 2023
bibliography: paper.bib


---

# Summary

MapReader is an interdisciplinary software library for processing digitized maps [@Hosseini_mapreader] and other types of images by 'patching' them into small, custom-sized cells which are then classified according to the user's needs. MapReader thus offers a flexible pipeline which can be used both for manual annotation of small datasets as well as for computer-vision-based inference of large collections. As an example, in @Hosseini_mapreader, we utilized MapReader's interface to manually annotate 62,020 patches, used its functionalities to train a suite of computer vision models and performed model inference on approximately 30.5 million patches.

MapReader's approach was inspired by methods in biomedical imaging and adapted for use by historians, and it is suitable for a wide range of applications in image analysis: it has, for example, been applied to an image classification problem in plant phenotype research [@Corcoran]. This cross-pollination between the humanities and the natural sciences was made possible by the open and reproducible research methods at the heart of MapReader.

MapReader pioneers a methodological shift in how historians interact with maps as primary sources. Sustained engagement with big collections of maps rarely moves beyond analysis of cartographic history. To change this, MapReader encourages historians to reflect on the content of maps and is designed to facilitate linking datasets representing visual map content with other historical geospatial data.

In this paper, we present the MapReader release at the conclusion of the Living with Machines project, which supported the development of the software and associated historical research. This release represents the culmination of extensive work to improve MapReader's usability, especially through clear documentation and tutorials.

![MapReader modules and input-outputs. Credit: Rosie Wood.\label{fig:modules}](https://hackmd.io/_uploads/HJWJatQEa.png)


# Statement of need

Since the 1990s, map libraries have been scanning maps and creating digital collections of these images [@Hosseini_maps]. As of 2023, there are more than a million images of maps in digital libraries and archives around the world, and yet it is very difficult for anyone to do more than browse them in a web viewer.

MapReader makes it possible to ask questions of thousands of digitized maps at a time, a fundamentally different intellectual experience from both the traditional manner of viewing a few maps at a time on a reading room table as well as the act of visually scanning digital files sequentially. As an example, we used MapReader to process a collection of approximately 16,000 nineteenth-century Ordnance Survey map sheets (approximately 30.5 million patches) covering England, Wales and Scotland [@Hosseini_mapreader]. Inspired by the possibility of seeing a series of maps stitched together in seamless layers---such as the National Library of Scotland's Ordnance Survey map viewing interface---MapReader takes the next step by transforming the experience of working with maps from surface exploration to critical investigation [@Hosseini_maps].


# Related Work

MapReader is among the first end-to-end pipelines for processing historical maps and other images that are designed to lower barriers to experimenting with computer vision in answering research questions about large image datasets. Other projects are emerging which perform similar research tasks with the visual content in historical map collections [@Petitpierre; @Combes]. Other tools, like the Distant Viewing Toolkit [@Arnold], address similar needs for other kinds of media. As part of a collaboration between Machines Reading Maps and the David Rumsey Historical Map Collection, the Knowledge Computing Lab released mapKurator [@mapkurator], which takes map image input and creates geojson files of all text on each map.


# Documentation

MapReader aims to build computational skills among historians. Our extensive work on documentation and training, including substantial updates to MapReader since @Hosseini_mapreader, reflect this commitment. As historians explore the possibilities of computational methods for novel historical research, MapReader models how computational tools can unlock difficult-to-use primary sources and how we can embrace open research practices as a way to encourage learning. We welcome contributions and requests for new documentation or tutorials.

Our documentation includes:
- About MapReader: A basic introduction to the software and its origins
- Events: Activities where the community can engage with MapReader
- Project Curriculum Vitae: Papers, talks, workshops, etc. delivered by the MapReader team
- Coding Basics: Skills and tools that are useful for getting started with MapReader
- Installation Instructions: How to install MapReader
- Input Guidance: What kind of maps and which formats work well in MapReader
- User Guide: Walkthrough of how to run MapReader
- Worked Examples: Jupyter notebooks demonstrating uses of MapReader for specific cases (with data, e.g. @Hosseini_mapreader_data)
- API Reference: Auto-generated API reference documentation
- Code of Conduct and Inclusivity: Our approach to ethical and inclusive conduct
- Contribution Guide: How to engage with MapReader
- Developer's Guide: How to update the MapReader version number and upload to package managers

# Conclusion

Through its conceptual approach, modular structure, documentation, and worked examples, MapReader enables researchers to ask questions of large collections of maps. It represents a novel approach to digitizing map content, one which intentionally prevents the collection of overly precise data from cartographic documents. MapReader embraces a humanistic approach to data creation and curation, offering an alternative or complement to pixel-level image segmentation.

# Acknowledgements

This work was supported by Data/Culture (AHRC grant AH/Y00745X/1), Living with Machines (AHRC grant AH/S01179X/1), and The Alan Turing Institute (EPSRC grant EP/N510129/1). Living with Machines, funded by the UK Research and Innovation (UKRI) Strategic Priority Fund, was a multidisciplinary collaboration delivered by the Arts and Humanities Research Council (AHRC), with The Alan Turing Institute, the British Library and the Universities of Cambridge, East Anglia, Exeter, and Queen Mary University of London. Maps and their metadata in MapReader are reproduced with the permission of the National Library of Scotland (https://maps.nls.uk/index.html). We also wish to thank participants in events in 2023 who provided feedback on using MapReader.

# Contribution Statement

Katherine McDonough wrote and revised this article, with substantial contributions from Daniel C.S. Wilson and Rosie Wood. Andy Smith, Kalle Westerling, Kaspar Beelen and Kasra Hosseini reviewed the final manuscript. Please see contributions to the MapReader software library at https://github.com/Living-with-machines/MapReader#contributors, including work from all named authors.

# References
