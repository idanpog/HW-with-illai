import java.net.*;
import java.io.*;

public class Sender extends Thread {
    private int port = -1;
    private Socket socket = null;
    private DataOutputStream outputStream = null;

    public Sender(int port)
    {
        this.port = port;
    }

    public void start()
    {
        while(true){
            try {
                assert this.port != -1;
                Socket socket = new Socket("localhost", this.port);
                this.outputStream = new DataOutputStream(socket.getOutputStream());
                System.out.println("Sender started an output stream on port " + port);
                break;

            } catch (IOException e) {
                System.out.println("Error in Sender with port " + this.port+ ": " + e);
                System.out.println("Trying again...");
            }
        }
    }
    public void send(String message)
    {
        try {
            // Send the message
            outputStream.writeUTF(message);
        } catch (IOException e) {
            System.out.println("Error: " + e);
        }
    }
    public void close()
    {
        try {
            // Close the socket
            outputStream.close();
            socket.close();
        } catch (IOException e) {
            System.out.println("Error: " + e);
        }
    }
    public String toString(){
        return "Sender on port " + this.port;
    }
}

