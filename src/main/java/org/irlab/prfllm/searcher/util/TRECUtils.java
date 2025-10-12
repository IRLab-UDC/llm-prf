package org.irlab.prfllm.searcher.util;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.lucene.document.Document;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;

/**
 * Utility class for TREC-related operations such as parsing topics and writing run files.
 */
public class TRECUtils {

  /**
   * Topic class for TREC topics
   */
  public static class Topic {
    public String num;
    public String title;
    public String description;
  }

  /**
   * Parse TREC topics file in standard TREC format.
   * 
   * @param topicsPath Path to the TREC topics file
   * @return List of parsed topics
   * @throws IOException If file cannot be read
   */
  public static List<Topic> parseTRECTopics(String topicsPath) throws IOException {
    List<Topic> topics = new ArrayList<>();
    BufferedReader br = new BufferedReader(new FileReader(topicsPath));
    String line;
    Topic topic = null;
    while ((line = br.readLine()) != null) {
      if (line.trim().startsWith("<num>")) {
        topic = new Topic();
        topic.num = line.replaceAll("[^0-9]", "");
      } else if (line.trim().startsWith("<title>")) {
        // We also remove the Topic: prefix if present
        if (topic != null)
          topic.title = line.replace("<title>", "").replace("Topic:", "").trim();

      } else if (line.trim().startsWith("<desc>")) {
        if (topic != null)
          topic.description = br.readLine().replace("Description:", "").trim();
      } else if (line.trim().startsWith("</top>")) {
        if (topic != null)
          topics.add(topic);
      }
    }
    br.close();
    return topics;
  }

  /**
   * Write results to TREC run file format.
   * 
   * @param writer BufferedWriter for output file
   * @param qid Query ID
   * @param results TopDocs containing search results
   * @param searcher IndexSearcher to retrieve document fields
   * @param runTag Tag to identify the run (full tag for rank 1, "--" for others to save space)
   * @throws IOException If writing fails
   */
  public static void writeTrecRun(BufferedWriter writer, String qid, TopDocs results, IndexSearcher searcher,
      String runTag) throws IOException {
    int rank = 1;
    for (ScoreDoc sd : results.scoreDocs) {
      Document doc = searcher.storedFields().document(sd.doc);
      String docno = doc.get("DOCNO");
      // Write full runTag for rank 1, use "--" for the rest to save space
      String tag = (rank == 1) ? runTag : "--";
      writer.write(String.format("%s Q0 %s %d %f %s\n", qid, docno, rank, sd.score, tag));
      rank++;
    }
  }
}
