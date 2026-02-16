package org.irlab.ecir26.searcher.util;

import java.util.*;
import java.util.Map.Entry;

public class PutHashMapInOrder {

  public static <K, V extends Comparable<? super V>> Map<K, V> sort(Map<K, V> map) {

    List<Entry<K, V>> list = new ArrayList<>(map.entrySet());
    list.sort(Entry.comparingByValue());
    Collections.reverse(list);

    Map<K, V> result = new LinkedHashMap<>();
    for (Entry<K, V> entry : list) {
      result.put(entry.getKey(), entry.getValue());
    }

    return result;
  }
}
