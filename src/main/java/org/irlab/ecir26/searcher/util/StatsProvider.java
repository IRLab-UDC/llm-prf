package org.irlab.ecir26.searcher.util;

import org.apache.commons.lang3.tuple.Pair;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.lucene.index.*;
import org.apache.lucene.util.BytesRef;

import java.io.IOException;
import java.util.concurrent.ConcurrentHashMap;

public final class StatsProvider {

  private static final Logger LOG = LogManager.getLogger(StatsProvider.class);
  private final IndexReader reader;
  private final ConcurrentHashMap<String, Long> cacheLexiconSize;
  private final ConcurrentHashMap<Pair<Integer, String>, Terms> cacheTermVector;
  private final ConcurrentHashMap<Integer, ConcurrentHashMap<String, Integer>> cacheDocTerm;
  private TermVectors termvectors;

  public StatsProvider(IndexReader reader) {

    this.reader = reader;
    try {
      this.termvectors = reader.termVectors();
    } catch (IOException e) {
      e.printStackTrace();
      LOG.error(e.getMessage());
      throw new RuntimeException(e);
    }
    this.cacheLexiconSize = new ConcurrentHashMap<>();
    this.cacheTermVector = new ConcurrentHashMap<>();
    this.cacheDocTerm = new ConcurrentHashMap<>();
  }

  public int getTermFrequency(String term, int doc, String field) {

    try {

      final BytesRef termBytes = new Term(field, term).bytes();

      if (cacheDocTerm.containsKey(doc)) {

        ConcurrentHashMap<String, Integer> cachedBytesRef = cacheDocTerm.get(doc);

        if (cachedBytesRef.containsKey(termBytes.utf8ToString())) {
          return cachedBytesRef.get(termBytes.utf8ToString());
        } else {
          return 0;
        }

      } else {

        Terms termVector;
        Pair<Integer, String> key = Pair.of(doc, field);
        ConcurrentHashMap<String, Integer> cachedBytesRef = new ConcurrentHashMap<>();

        if (cacheTermVector.containsKey(key)) {
          termVector = cacheTermVector.get(key);
        } else {
          termVector = getTermVector(doc, field);
          cacheTermVector.put(key, termVector);
        }

        if (termVector == null) {

          return 0;
        }

        TermsEnum termsEnum = termVector.iterator();
        BytesRef otherTermBytes;

        while ((otherTermBytes = termsEnum.next()) != null) {
          PostingsEnum termPosting = termsEnum.postings(null);
          termPosting.nextDoc();
          cachedBytesRef.put(otherTermBytes.utf8ToString(), termPosting.freq());
        }
        cacheDocTerm.put(doc, cachedBytesRef);

        if (cachedBytesRef.containsKey(termBytes.utf8ToString())) {
          return cachedBytesRef.get(termBytes.utf8ToString());
        } else {
          return 0;
        }
      }

    } catch (final IOException e) {

      LOG.error(e.getMessage());
      return 0;
    }

  }

  public long getDocTokensSize(int doc, String field) {

    try {

      Terms termVector = getTermVector(doc, field);

      if (termVector == null) {

        return 0;
      }

      return termVector.getSumTotalTermFreq();
    } catch (final IOException e) {

      LOG.error(e.getMessage());
      throw new RuntimeException(e);
    }
  }

  public long getDocTermSize(int doc, String field) {

    try {

      Terms termVector = getTermVector(doc, field);

      if (termVector == null) {

        return 0;
      }

      return termVector.size();
    } catch (final IOException e) {

      LOG.error(e.getMessage());
      throw new RuntimeException(e);
    }
  }


  public long getTotalTermFrequency(String term, String field) {

    try {

      return reader.totalTermFreq(new Term(field, term));
    } catch (IOException e) {

      LOG.error(e.getMessage());
      throw new RuntimeException(e);
    }
  }

  public long getCollectionTokensSize(String field) {

    try {

      return reader.getSumTotalTermFreq(field);
    } catch (IOException e) {

      LOG.error(e.getMessage());
      throw new RuntimeException(e);
    }
  }


  public long getCollectionLexiconSize(String field) {

    if (cacheLexiconSize.containsKey(field)) {

      return cacheLexiconSize.get(field);
    }

    try {
      Terms lexicon = MultiTerms.getTerms(reader, field);
      long termsCount = lexicon.size();

      if (termsCount != -1) {

        cacheLexiconSize.put(field, termsCount);
        return termsCount;
      }

      termsCount = 0;
      TermsEnum termsEnum = lexicon.iterator();

      while (termsEnum.next() != null) {

        termsCount = termsCount + 1;
      }

      cacheLexiconSize.put(field, termsCount);
      return termsCount;
    } catch (IOException e) {

      LOG.error(e.getMessage());
      throw new RuntimeException(e);
    }
  }

  public Terms getTermVector(int doc, String field) {

    try {

      return termvectors.get(doc, field);
    } catch (final IOException e) {

      LOG.error(e.getMessage());
      throw new RuntimeException(e);
    }
  }


}