from torch.utils.data import IterableDataset, get_worker_info

import neo4j

from typing import Callable
import logging as log

from .query import Neo4jQuery
from .iterator import Neo4jQueryIterator


class Neo4jDataset(IterableDataset):
    def __init__(self, database: neo4j.Driver, query: Neo4jQuery, fetch_size: int, transform: Callable = None):
        if not isinstance(database, neo4j.Driver):
            raise ValueError("Non neo4j database passes as argument")
        
        self.database = database
        self.query = query
        self.fetch_size = fetch_size
        self.length = None
        self.transform = transform
        self._query_total_length()

    def _query_total_length(self):
        log.info("Querying dataset length. It may take a while...")
        count_query_string = self.query.count()
        count_key = self.query.count_key
        session = self.database.session()
        result = session.run(count_query_string)
        values = result.value(count_key)
        if len(values) == 0:
            raise ValueError(f"Count key {count_key} not present!")
        
        count = int(values[0])
        log.info(f"Found total query length of {count}")
        self.length = count

    def __len__(self):
        return self.length

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            # single process data loading 
            start_position = 0
            step = self.fetch_size
            limit_per_step = self.fetch_size
        else:
            # multi process data loading
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            limit_per_step = self.fetch_size
            start_position = worker_id * limit_per_step
            step = num_workers * limit_per_step

        return Neo4jQueryIterator(
            self.database, self.query, self.transform,
            start_position, step, limit_per_step, self.length
        )
