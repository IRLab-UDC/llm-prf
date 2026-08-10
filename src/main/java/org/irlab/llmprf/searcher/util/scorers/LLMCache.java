package org.irlab.llmprf.searcher.util.scorers;

import java.io.IOException;

public interface LLMCache {
  LLMResult get(int queryId, int docId, String queryText, String narrative, String docText) throws IOException;

  boolean isEmpty();

  default boolean containsQuery(int queryId) {
    return false;
  }

  void close() throws IOException;
}
