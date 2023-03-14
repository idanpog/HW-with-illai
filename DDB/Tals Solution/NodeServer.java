import java.io.*;
import java.util.*;
import java.net.*;

public class NodeServer extends Thread{
    private int port; //port number
    private final Boolean[] listened; //list of nodes that have been listened to
    private int id; //id of the node
    private int source; //id of the node that sent the message
    private Double[][] matrix; //matrix of the node
    private NodeClient[] clients; //list of clients that will be used to send messages to other nodes

    /**
     * @param port the port to listen on
     * @param listened the array of nodes that have been listened to
     * @param id the id of the node
     * @param source the source of the message
     * @param matrix the matrix of the node
     * @param clients the list of clients that will be used to send messages to other nodes
     */
    public NodeServer(int port, Boolean[] listened, Double[][] matrix, NodeClient[] clients, int id, int source)
    {

        this.port = port;
        this.id = id;
        this.source = source;
        this.listened = listened;
        this.matrix = matrix;
        this.clients = clients;
    }

    /**
     * @ param array the array of pairs that will be converted to a string
     */
    private boolean containsFalse(Boolean[] array)
    {
        synchronized (array) {
            for (boolean b : array) {
                if (!b) {
                    return true;
                }
            }
            return false;
        }
    }
    @Override
    public void run() {
        super.run();
        ServerSocket serverSocket = null;
        try {
            serverSocket = new ServerSocket(this.port);
        } catch (IOException e) {
            throw new RuntimeException(e);
        } //create a server socket
        boolean connected = false;
        while (!connected) {
            try (
                    Socket clientSocket = serverSocket.accept();
                    ObjectInputStream in = new ObjectInputStream(new BufferedInputStream(clientSocket.getInputStream()))
            ) {
                connected = true;
                Pair<Pair<Integer, Integer>, Double>[] input; //the input from the client
                while (containsFalse(this.listened)) {
                    try {
                        input = (Pair<Pair<Integer, Integer>, Double>[]) in.readObject();
                        int source_id = input[0].getKey().getKey() - 1;
                        synchronized (this.listened) { // synchronize the listened array
                            if (!this.listened[source_id]) {
                                for (Pair<Pair<Integer, Integer>, Double> p : input) {
                                    int dest_id = p.getKey().getValue() - 1;
                                    this.matrix[source_id][dest_id] = p.getValue();
                                }
                                for (NodeClient client : this.clients) {
                                    client.queueMessage(input);
                                }
                                this.listened[source_id] = true;
                            }
                        }
                    } catch (IOException ignored) {

                    } catch (ClassNotFoundException | InterruptedException ex) {
                        throw new RuntimeException(ex);
                    }
                }
                serverSocket.close();
            } catch (IOException e) {
                throw new RuntimeException(e); //throw an exception if there is an error
            }

        }
    }
}
