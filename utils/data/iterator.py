import neo4j

from collections.abc import Iterator
from typing import Callable

from .query import Neo4jQuery

class Neo4jQueryIterator(Iterator):
    def __init__(
        self, database: neo4j.Driver, query: Neo4jQuery, transform: Callable,
        start_position: int, step: int, limit_per_step: int, total_length: int
    ):
        if not isinstance(database, neo4j.Driver):
            raise ValueError("Non neo4j driver passed as argument")

        self.start_position = start_position
        self.step = step
        self.limit_per_step = limit_per_step
        self.total_length = total_length
        self.position = start_position
        self.local_position = 0
        self.transform = transform
        self.query = query
        self.session = database.session()
        self.cache = None

    def __iter__(self):
        return self

    def __next__(self):
        if self.total_length is not None and self.position >= self.total_length:
            raise StopIteration()

        if self.cache is None or self.local_position >= len(self.cache):
            query_string = self.query.full(limit=self.limit_per_step, skip=self.position + self.step)
            results = self.session.run(query_string).data()
            if len(results) == 0:
                raise StopIteration()

            self.position += self.step
            self.local_position = 0
            self.cache = results
        
        item = self.cache[self.local_position]
        if self.transform is not None:
            item = self.transform(item)
        self.local_position += 1 

        return item       
