pagexml-hf
==========

Python package for preprocessing PageXML and image exports (e.g. from Transkribus or eScriptorium).
You can provide a ZIP export file or a Huggingface Hub repo containing the data to preprocess and upload to the Huggingface Hub.
This package is developed for the `Flow Project <https://flow-project.net>`_.

.. toctree::
   :maxdepth: 3
   :caption: Contents:

API Reference
=============
Main Package
------------
.. autosummary::
   :toctree: _autosummary
   :recursive:

   pagexml_hf

Core Modules
------------
Converter
~~~~~~~~~
.. autosummary::
   :toctree: _autosummary

   pagexml_hf.converter.XmlConverter

Parser
~~~~~~
.. autosummary::
   :toctree: _autosummary

   pagexml_hf.parser.XmlParser
   pagexml_hf.parser.TextLine

Exporters
~~~~~~~~~
.. autosummary::
   :toctree: _autosummary

   pagexml_hf.exporters.BaseExporter
   pagexml_hf.exporters.RawXMLExporter
   pagexml_hf.exporters.LineExporter
   pagexml_hf.exporters.RegionExporter
   pagexml_hf.exporters.TextExporter
   pagexml_hf.exporters.WindowExporter

Image Utils
~~~~~~~~~~~
.. autosummary::
   :toctree: _autosummary

   pagexml_hf.imageutils.ImageProcessor

Hub Utils
~~~~~~~~~
.. autosummary::
   :toctree: _autosummary

   pagexml_hf.hub_utils.HubUploader
   pagexml_hf.hub_utils.ProjectGrouper
   pagexml_hf.hub_utils.ReadmeGenerator
   pagexml_hf.hub_utils.ReadmeParser
   pagexml_hf.hub_utils.YamlGenerator
   pagexml_hf.hub_utils.FeatureYamlGenerator

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
