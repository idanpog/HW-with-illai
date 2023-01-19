import java.util.*;

public class LSP{

    public int seq_num;
    public int source_id;
    public Set<Pair<Pair<Integer, Integer>, Double>> l_v;

    public LSP(int seq_num, int source_id, Set<Pair<Pair<Integer, Integer>, Double>> l_v) {
        this.seq_num = seq_num;
        this.source_id = source_id;
        this.l_v = l_v;
    }

    public LSP(String message) {
        // Parse message
        String[] items = message.split("\t");
        this.seq_num = Integer.parseInt(items[0]);
        this.source_id = Integer.parseInt(items[1]);
        this.l_v = new HashSet<Pair<Pair<Integer, Integer>, Double>>();

        for (int i = 2; i < items.length; i++) {
            String[] item = items[i].split(" ");
            int id1 = Integer.parseInt(item[0]);
            int id2 = Integer.parseInt(item[1]);
            double weight = Double.parseDouble(item[2]);
            this.l_v.add(new Pair<Pair<Integer, Integer>, Double>(new Pair<Integer, Integer>(id1, id2), weight));
        }
    }

    public Set<Pair<Pair<Integer, Integer>, Double>> get_lv() {
        return this.l_v;
    }

    public int get_seq_num() {
        return this.seq_num;
    }

    public int get_source_id() {
        return this.source_id;
    }

    public String toString() {
        // Return string representation of LSP
        String output = this.seq_num + "\t" + this.source_id;
        for (Pair<Pair<Integer, Integer>, Double> item : this.l_v) {
            output += "\t" + item.getKey().getKey() + " " + item.getKey().getValue() + " " + item.getValue();
        }
        return output;
    }
}

