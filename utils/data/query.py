class Neo4jQuery(object):
    def __init__(self, match_clause: str, return_clause: str):
        self.match_clause = match_clause
        self.return_clause = return_clause

    def limit_clause(self, limit: int = None, skip: int = None):
        clauses = []
        if skip is not None:
            clauses += [f"SKIP {skip}"]
        if limit is not None:
            clauses += [f"LIMIT {limit}"]
        return " ".join(clauses)

    @property
    def count_key(self):
        return "count(*)"

    def count(self):
        clauses = [
            self.match_clause,
            "RETURN count(*)",
        ]

        return " ".join(clauses) + ";"

    def full(self, limit: int = None, skip: int = None):
        clauses = [
            self.match_clause,
            self.return_clause,
            self.limit_clause(limit, skip)
        ]

        return " ".join(clauses) + ";"