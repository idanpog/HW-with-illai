import java.net.*;
import java.util.*;
import java.io.*;


public class ExManager {
    private String path;
    private Hashtable<Integer, Node>  nodes_dict;
    private int num_of_nodes;
    private boolean connected = false;
    private Hashtable<Integer,Node> port2receiverNode;
    private Hashtable<Integer,Node> port2senderNode;

    public ExManager(String path){
        this.path = path;
        this.nodes_dict = new Hashtable<Integer, Node>();

        this.port2receiverNode = new Hashtable<Integer, Node>();
        this.port2senderNode = new Hashtable<Integer, Node>();
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
                Node node = new Node(line, this.num_of_nodes);
                this.nodes_dict.put(Integer.parseInt(line.split(" ")[0]), node);
                this.update_dicts(line, node);
            }

            scanner.close();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }
    private void update_dicts(String line, Node node){
        /**
         * builds the dictionaries that map the ports to their corresponding nodes
         * port2receiverNode - maps the port that a node listens to the node
         * port2senderNode - maps the port that a node sends to the node
         */
        String[] data = line.split(" ");
        for (int i = 1; i < data.length; i+=4) {
            int send_port = Integer.parseInt(data[i+2]);
            int listen_port = Integer.parseInt(data[i+3]);
            this.port2receiverNode.put(listen_port, node);
            this.port2senderNode.put(send_port, node);
        }
    }
    private void initiate_connections(){
        /**
         * starts the threads of all the nodes
         */
        this.port2receiverNode.forEach((port, node) -> {
            Receiver r = new Receiver(port);
            Sender s = new Sender(port);
            r.bind();
            s.start();
            r.start();
            this.port2receiverNode.get(port).append_receiver(r);
            this.port2senderNode.get(port).append_sender(s);
        });
    }
    private void refresh_nodes(){
        /**
         * refreshes the nodes they could be runned again.
         */
        this.nodes_dict.forEach((id, node) -> {
            nodes_dict.put(id, node.refreshed());
        });
    }
    public void start(){

        //wait for 1 seconds using synchronized
        synchronized (this) {
            try {
                wait(1000);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }

        if (this.connected==false){
            this.initiate_connections();
            this.connected = true;
            //wait for all the nodes to finish their initialization
            try {
                Thread.sleep(500);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }

        this.refresh_nodes();
        this.nodes_dict.values().parallelStream().forEach(Node::start);


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
            node.halt_loop();
        }
        for (Node node : this.nodes_dict.values()) {
            node.terminate();
        }

        // wait for all nodes to join
        for (Thread Node : this.nodes_dict.values()) {
            try {
                Node.join();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
    }
}
