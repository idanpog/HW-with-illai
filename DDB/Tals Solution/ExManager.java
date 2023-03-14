import java.io.*;
import java.util.*;
import java.net.*;
import java.util.concurrent.*;


public class ExManager {
    private String path; //path to the file
    private int numOfNodes; //number of nodes in the network
    private Node[] nodes; //list of nodes in the network
    private CountDownLatch latch; //signal that will be used to wait for all threads to finish
    ExecutorService threadPool; //thread pool that will be used to run the threads
    public ExManager(String path){
        this.path = path;
    }


    public int getNum_of_nodes() {
        return this.numOfNodes;
    }

    public void update_edge(int id1, int id2, double weight){
        nodes[id1-1].updateWeight(id2, weight);
        nodes[id2-1].updateWeight(id1, weight);
    }

    public void read_txt() throws FileNotFoundException {
        // Read the file and create the nodes

        // Read file:
        Scanner scanner = new Scanner(new File(this.path));

        boolean first_line = true;
        while(scanner.hasNextLine()){
            String line = scanner.nextLine(); //read line
            if (line.equals("stop")) { //stop reading the file
                break;
            }
            if (first_line){
                this.numOfNodes = Integer.parseInt(line);
                this.latch = new CountDownLatch(this.numOfNodes);
                this.nodes = new Node[this.numOfNodes];
                this.threadPool = Executors.newFixedThreadPool(this.numOfNodes);
                //create a thread pool with the number of nodes
                first_line = false;
            }
            else{
                String[] data = line.split(" "); //split the line into the data
                int id = Integer.parseInt(data[0]);
                Node n = new Node(id, this.numOfNodes, this.latch);
                int i = 1;
                int data_len = data.length;
                while (i < data_len){
                    int neighborId = Integer.parseInt(data[i]);
                    double weight = Double.parseDouble(data[i+1]);
                    int sendPort = Integer.parseInt(data[i+2]);
                    int listenPort = Integer.parseInt(data[i+3]);
                    n.addNeighbor(neighborId, weight, sendPort, listenPort);
                    i += 4;
                }
                this.nodes[id-1] = n;
            }
        }
    }


    public void start() throws InterruptedException {
        // Start the nodes
        CountDownLatch latch = new CountDownLatch(this.numOfNodes);
        for (int i = 0; i < this.numOfNodes; i++){
            this.nodes[i].setLatch(latch);
            this.threadPool.execute(this.nodes[i]);
        }
        try {
            latch.await();
        } catch (InterruptedException E) {
            // handle
        }

    }

    public void terminate(){
        // Terminate the nodes
        for (int i = 0; i < this.numOfNodes; i++){
            this.nodes[i].terminate(); //terminate the node
        }
        this.threadPool.shutdown(); //shutdown the thread pool

    }

    public Node get_node(int i) {
        return this.nodes[i-1];
    }
}
