package org.irlab.prfllm.searcher.util.scorers;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URI;
import java.net.URL;

/**
 * Helper to call Ollama API for document relevance assessment.
 * Expects Ollama running on localhost with a model that can answer yes/no questions.
 */
public class OllamaScorer {

    private static final String SERVICE_URL = "http://namo:11434/api/generate";
    private static final String DEFAULT_MODEL = "llama3.1:8b-instruct-fp16";
    private static String model = DEFAULT_MODEL;

    /**
     * Result from Ollama evaluation
     */
    public static class OllamaResult {
        public final boolean isRelevant;
        public final String response;
        
        public OllamaResult(boolean isRelevant, String response) {
            this.isRelevant = isRelevant;
            this.response = response;
        }
    }

    /**
     * Set the Ollama model to use
     */
    public static void setModel(String modelName) {
        model = modelName;
    }

    /**
     * Evaluate document relevance using Ollama
     */
    public static OllamaResult evaluate(String query, String document) {
        try {
            URL url = URI.create(SERVICE_URL).toURL();
            HttpURLConnection con = (HttpURLConnection) url.openConnection();
            con.setRequestMethod("POST");
            con.setRequestProperty("Content-Type", "application/json; charset=UTF-8");
            con.setDoOutput(true);
            con.setConnectTimeout(5000);
            con.setReadTimeout(120000); // 2 minutes for LLM response

            // Truncate document to avoid context limits
            String truncatedDoc = document.length() > 2000 ? document.substring(0, 2000) : document;
            
            // Create prompt for yes/no relevance judgment
            String prompt = String.format(
                "Given the following query and document, determine if the document is relevant to the query.\n\n" +
                "Query: %s\n\n" +
                "Document: %s",
                query, truncatedDoc
            );

            // JSON schema for structured output
            String schema = "{" +
                "\"type\": \"object\"," +
                "\"properties\": {" +
                    "\"relevance\": {\"type\": \"boolean\"}" +
                "}," +
                "\"required\": [\"relevance\"]" +
            "}";

            // Build Ollama JSON request with format parameter
            String payload = String.format(
                "{\"model\": %s, \"prompt\": %s, \"stream\": false, \"format\": %s, \"options\": {\"temperature\": 0.0}}",
                jsonEscapeAndQuote(model),
                jsonEscapeAndQuote(prompt),
                schema
            );

            try (OutputStream os = con.getOutputStream()) {
                byte[] input = payload.getBytes("utf-8");
                os.write(input, 0, input.length);
            }

            int code = con.getResponseCode();
            InputStreamReader isr = new InputStreamReader(
                    code >= 200 && code < 300 ? con.getInputStream() : con.getErrorStream(), "utf-8");
            try (BufferedReader br = new BufferedReader(isr)) {
                StringBuilder resp = new StringBuilder();
                String line;
                while ((line = br.readLine()) != null) {
                    resp.append(line);
                }
                return parseResponse(resp.toString());
            }
        } catch (Exception e) {
            System.err.println("OllamaScorer error: " + e.getMessage());
            e.printStackTrace();
            return new OllamaResult(false, "ERROR: " + e.getMessage());
        }
    }
    
    /**
     * Convenience method: returns only boolean relevance
     */
    public static boolean isRelevant(String query, String document) {
        return evaluate(query, document).isRelevant;
    }

    private static String jsonEscapeAndQuote(String s) {
        if (s == null) return "\"\"";
        StringBuilder sb = new StringBuilder();
        sb.append('"');
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            switch (c) {
                case '\\': sb.append("\\\\"); break;
                case '"': sb.append("\\\""); break;
                case '\b': sb.append("\\b"); break;
                case '\f': sb.append("\\f"); break;
                case '\n': sb.append("\\n"); break;
                case '\r': sb.append("\\r"); break;
                case '\t': sb.append("\\t"); break;
                default:
                    if (c < ' ') {
                        sb.append(String.format("\\u%04x", (int) c));
                    } else {
                        sb.append(c);
                    }
            }
        }
        sb.append('"');
        return sb.toString();
    }

    // Parse Ollama JSON response to extract the relevance value
    private static OllamaResult parseResponse(String json) {
        if (json == null || json.isEmpty()) {
            return new OllamaResult(false, "");
        }
        
        // First, extract the "response" field from Ollama's wrapper JSON
        String responseKey = "\"response\"";
        int idx = json.indexOf(responseKey);
        if (idx == -1) {
            return new OllamaResult(false, "");
        }
        
        int colon = json.indexOf(':', idx + responseKey.length());
        if (colon == -1) {
            return new OllamaResult(false, "");
        }
        
        int i = colon + 1;
        while (i < json.length() && Character.isWhitespace(json.charAt(i))) i++;
        
        if (i >= json.length() || json.charAt(i) != '"') {
            return new OllamaResult(false, "");
        }
        
        i++; // skip opening quote
        int start = i;
        
        // read until closing quote, handling escape sequences
        while (i < json.length()) {
            if (json.charAt(i) == '\\' && i + 1 < json.length()) {
                i += 2;
                continue;
            }
            if (json.charAt(i) == '"') {
                break;
            }
            i++;
        }
        
        if (i >= json.length()) {
            return new OllamaResult(false, "");
        }
        
        // Unescape the response content
        String escapedResponse = json.substring(start, i);
        String responseContent = unescapeJson(escapedResponse);
        
        // Clean up whitespace and control characters
        responseContent = responseContent.replaceAll("[\\n\\r\\t]+", " ").trim();
        
        // Now parse the structured JSON within the response: {"relevance": true/false}
        // Look for "relevance" key (with or without escaped quotes)
        String relevanceKey = "\"relevance\"";
        int relevIdx = responseContent.indexOf(relevanceKey);
        if (relevIdx == -1) {
            // Try without quotes in case format is different
            relevIdx = responseContent.indexOf("relevance");
            if (relevIdx == -1) {
                return new OllamaResult(false, responseContent);
            }
        }
        
        int relevColon = responseContent.indexOf(':', relevIdx);
        if (relevColon == -1) {
            return new OllamaResult(false, responseContent);
        }
        
        int j = relevColon + 1;
        while (j < responseContent.length() && Character.isWhitespace(responseContent.charAt(j))) j++;
        
        // Read boolean value
        boolean isRelevant = false;
        if (j < responseContent.length()) {
            String remaining = responseContent.substring(j).toLowerCase().trim();
            isRelevant = remaining.startsWith("true");
        }
        
        return new OllamaResult(isRelevant, responseContent);
    }
    
    // Unescape JSON string (handle \", \\, \n, \r, \t, etc.)
    private static String unescapeJson(String s) {
        if (s == null) return "";
        StringBuilder sb = new StringBuilder();
        int i = 0;
        while (i < s.length()) {
            if (s.charAt(i) == '\\' && i + 1 < s.length()) {
                char next = s.charAt(i + 1);
                switch (next) {
                    case '"': sb.append('"'); break;
                    case '\\': sb.append('\\'); break;
                    case 'n': sb.append('\n'); break;
                    case 'r': sb.append('\r'); break;
                    case 't': sb.append('\t'); break;
                    case 'b': sb.append('\b'); break;
                    case 'f': sb.append('\f'); break;
                    case 'u': // Unicode escape
                        if (i + 5 < s.length()) {
                            try {
                                int code = Integer.parseInt(s.substring(i + 2, i + 6), 16);
                                sb.append((char) code);
                                i += 4; // Will be incremented by 2 below
                            } catch (NumberFormatException e) {
                                sb.append('\\').append(next);
                            }
                        } else {
                            sb.append('\\').append(next);
                        }
                        break;
                    default:
                        sb.append('\\').append(next);
                }
                i += 2;
            } else {
                sb.append(s.charAt(i));
                i++;
            }
        }
        return sb.toString();
    }
}
