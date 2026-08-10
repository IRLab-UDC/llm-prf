package org.irlab.llmprf.searcher.util.scorers;

import java.io.IOException;
import java.util.List;

public interface TermsProvider {
  List<String> get(int queryId, int docId, String queryText, String narrative, String docText) throws IOException;
  boolean containsQuery(int queryId);
  void close() throws IOException;
}
