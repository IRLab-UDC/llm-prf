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
    public String narrative;
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
    StringBuilder narrative = null;
    boolean inNarrative = false;
    
    while ((line = br.readLine()) != null) {
      String trimmedLine = line.trim();
      
      // Check if we're entering a new tag section (exits narrative mode)
      if (trimmedLine.startsWith("<") && !trimmedLine.startsWith("</top>")) {
        // If we were in narrative mode, save it
        if (inNarrative && narrative != null && topic != null) {
          topic.narrative = narrative.toString().trim();
          inNarrative = false;
          narrative = null;
        }
      }
      
      if (trimmedLine.startsWith("<num>")) {
        topic = new Topic();
        topic.num = line.replaceAll("[^0-9]", "");
      } else if (trimmedLine.startsWith("<title>")) {
        // Remove the <title> tag and optional "Topic:" prefix
        if (topic != null)
          topic.title = line.replace("<title>", "").replace("Topic:", "").trim();
      } else if (trimmedLine.startsWith("<desc>")) {
        // Description can be on the same line or the next line
        String descText = line.replace("<desc>", "").replace("Description:", "").trim();
        if (descText.isEmpty()) {
          // Description is on the next line
          String nextLine = br.readLine();
          if (nextLine != null && topic != null) {
            topic.description = nextLine.replace("Description:", "").trim();
          }
        } else {
          // Description is on the same line
          if (topic != null)
            topic.description = descText;
        }
      } else if (trimmedLine.startsWith("<narr>")) {
        inNarrative = true;
        narrative = new StringBuilder();
        // Check if narrative text is on the same line
        String narrText = line.replace("<narr>", "").replace("Narrative:", "").trim();
        if (!narrText.isEmpty()) {
          narrative.append(narrText);
        }
        // If narrative is empty on this line, it will continue on following lines
      } else if (trimmedLine.startsWith("</top>")) {
        if (topic != null) {
          // Save narrative if we're still in narrative mode
          if (inNarrative && narrative != null) {
            topic.narrative = narrative.toString().trim();
            inNarrative = false;
            narrative = null;
          }
          topics.add(topic);
          topic = null;
        }
      } else if (inNarrative && !trimmedLine.isEmpty() && !trimmedLine.equals("Narrative:")) {
        // Continue accumulating narrative text (skip empty lines and standalone "Narrative:" labels)
        if (narrative.length() > 0) {
          narrative.append(" ");
        }
        narrative.append(trimmedLine);
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
