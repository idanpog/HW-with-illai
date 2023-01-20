import java.net.*;
import java.io.*;

public class Sender extends Thread {
    private int port = -1;
    private DataOutputStream outputStream = null;

    public Sender(int port)
    {
        this.port = port;
    }

    public void start()
    {
        int count = 0;
        while(count < 10) {
            try {
                assert this.port != -1;
//                System.out.println("Sender trying to connect to port " + port);
                Socket socket = new Socket("localhost", this.port);
//                System.out.println("Sender connected to port " + this.port);
                this.outputStream = new DataOutputStream(socket.getOutputStream());
//                System.out.println("Sender started an output stream on port " + port);
                break;

            } catch (IOException e) {
                System.out.println("Error in Sender with port " + this.port+ ": " + e);
                System.out.println("Trying again...");
            }
            count++;
            try {
                synchronized(this){
                wait(10);}
            } catch (InterruptedException e) {
                e.printStackTrace();
            }


        }
        if (count==10)
        {System.out.println("Failed to start Sender on port " + port);}
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
        } catch (IOException e) {
            System.out.println("Error: " + e);
        }
    }
    public String toString(){
        return "Sender on port " + this.port;
    }
}

