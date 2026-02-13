package org.irlab.ecir26.searcher.util;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class TRECUtils {

  public static class Topic {
    public String num;
    public String title;
    public String description;
    public String narrative;
  }

  public static List<Topic> parseTRECTopics(String topicsPath) {
    List<Topic> topics = new ArrayList<>();

    try (BufferedReader br = new BufferedReader(new FileReader(topicsPath))) {
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
          if (topic != null) topic.title = line.replace("<title>", "").replace("Topic:", "").trim();
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
            if (topic != null) topic.description = descText;
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
          if (!narrative.isEmpty()) {
            narrative.append(" ");
          }
          narrative.append(trimmedLine);
        }
      }
    } catch (IOException e) {
      throw new RuntimeException(e);
    }

    return topics;
  }
}
