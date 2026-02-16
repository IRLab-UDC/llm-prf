package org.irlab.ecir26.searcher.util.scorers;

import java.io.IOException;

/**
 * Common interface for LLM-based relevance judgment caches.
 * Implementations include MonoT5Cache and VLLMCache.
 */
public interface LLMCache {
  /**
   * Get or compute LLM relevance judgment for a query-document pair.
   *
   * @param queryId   Query identifier
   * @param docId     Document identifier
   * @param queryText Query text
   * @param narrative Narrative or instructions for the query (can be null)
   * @param docText   Document text
   * @return LLMResult containing relevance judgment and probabilities
   * @throws IOException if there's an error accessing the cache or LLM
   */
  LLMResult get(int queryId, int docId, String queryText, String narrative, String docText) throws IOException;

  /**
   * Check if the cache is empty (no entries).
   * Used to determine if sequential processing is needed to avoid overloading.
   *
   * @return true if cache has no entries, false otherwise
   */
  boolean isEmpty();

  /**
   * Close the cache and release resources.
   *
   * @throws IOException if there's an error closing the cache
   */
  void close() throws IOException;
}
