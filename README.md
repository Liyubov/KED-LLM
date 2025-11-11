# KED

This repository contains scripts for KED event clustering algorithm.  

**Main KED pipeline**

The main logic lives in `event_clustering_pipeline.py`.

Data requirements for the pipeline:

**Entities table**. For each extracted mention, include the columns entity (the original string), type (the entity category, e.g., object, toponym, actor, etc.), and chunk (the identifier of the text fragment). When building the graph, these fields are normalized and used both to assign canonical attributes to nodes and to link nodes to texts via chunk.

**Relations table.** Required columns are subject and object, which must match entity values from the entities list; predicate and source_text provide additional context; time is the event timestamp (used to slice the graph into time intervals); event_id is the target label for evaluating clustering quality.

**Text corpus**. The text column is mandatory; others (e.g., tweet_id, entities) may be present but are not required.

**Logic for optimal parameter detection**

Use `parallel_param_search.py` to run a grid search over Louvain, Leiden and
Node2Vec (with either k-means or HDBSCAN) configurations in parallel. The script
can toggle additional semantic, cross-type and geographic edges just like the
companion notebook. Semantic and geographic edges are injected after
temporal slicing so that only surviving nodes connect. A progress bar shows the
remaining configurations and all output is structured via the standard logging
module.

This script uses the same three files and assumes the presence of the listed entity types to configure semantic and geographic edges. The toponym cache is passed as cache_file in grid-search tasks.
