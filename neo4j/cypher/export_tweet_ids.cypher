// Export tweet ids

MATCH (t: Tweet) WITH t.id AS id RETURN id;
