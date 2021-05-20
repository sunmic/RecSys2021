// Export user ids

MATCH (u: User) WITH u.id AS id RETURN id;
