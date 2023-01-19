import java.net.*;
import java.util.*;
import java.io.*;


public class ExManager {
    private String path;
    private Hashtable<Integer, Node>  nodes_dict;
    private int num_of_nodes;
    private boolean connected = false;
    // your code here

    public ExManager(String path){
        this.path = path;
        this.nodes_dict = new Hashtable<Integer, Node>();
    }

    public Node get_node(int id){
        return this.nodes_dict.get(id);
    }

    public int getNum_of_nodes() {
        return this.num_of_nodes;
    }

    public void update_edge(int id1, int id2, double weight){
        //updates the weight of the edge between the nodes with id = id1 and id = id2
        this.nodes_dict.get(id1).update_neighbor_weight(id2, weight);
        this.nodes_dict.get(id2).update_neighbor_weight(id1, weight);
        this.nodes_dict.get(id1).start_broadcast();
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
                this.nodes_dict.put(Integer.parseInt(line.split(" ")[0]), new Node(line, this.num_of_nodes));
            }
            scanner.close();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }
    private void initiate_connections(){
        /**
         * starts the threads of all the nodes
         */
        this.nodes_dict.values().parallelStream().forEach(Node::launch_ports);
        this.nodes_dict.values().parallelStream().forEach(Node::start);
        //wait for all the nodes to start
        for (Thread Node : this.nodes_dict.values()) {
            try {
                Node.join();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
    }
    public void start(){

        if (this.connected==false){
            this.initiate_connections();
            this.connected = true;
        }
        this.nodes_dict.values().parallelStream().forEach(Node::run_link_state);
        // wait for all nodes to join
        for (Thread Node : this.nodes_dict.values()) {
            try {
                Node.join();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }


    }

    public void terminate(){
        for (Node node : this.nodes_dict.values()) {
            node.terminate();
        }
    }
}
