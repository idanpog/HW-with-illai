import java.net.*;
import java.util.*;
import java.io.*;
import Pair;

public class ExManager {
    private String path;
    private Hashtable<Integer, Node>  nodes_dict;
    private int num_of_nodes;
    // your code here

    public ExManager(String path){
        this.path = path;
        this.nodes_dict = new Hashtable<Integer, Node>();
    }

    public Node getNode(int id){
        return this.nodes_dict.get(id);
    }

    public int getNum_of_nodes() {
        return this.num_of_nodes;
    }

    public void update_edge(int id1, int id2, double weight){
    this.nodes_dict.get(id1).update_neighbour_weight(id2, weight);
    this.nodes_dict.get(id2).update_neighbour_weight(id1, weight);
    }


    public void read_txt(){
        /**
         * reads the input file and creates the nodes and edges
         */
        try {
            Scanner scanner = new Scanner(new File(this.path));
            this.num_of_nodes = Integer.parseInt(scanner.nextLine());
            while (scanner.hasNextLine()) {
                String line = scanner.nextLine();

                if (line.equals("stop")){
                    break;}
            }
            scanner.close();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }
    private Node parse_node(String line){
        /**
         * parses a line of the input file and creates a node
         */
        String[] data = line.split(" ");
        int id = Integer.parseInt(data[0]);
        for (int i = 1; i < data.length; i+=4) {
            int id2 = Integer.parseInt(data[i]);
            double weight = Integer.parseInt(data[i+1]);
            int send_port = Integer.parseInt(data[i+2]);
            int listen_port = Integer.parseInt(data[i+3]);
            Node node = new Node()
            // your code here
        }
    }
    public void start(){
        // your code here
    }

    public void terminate(){
        // your code here
    }
}
