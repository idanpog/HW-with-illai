import java.net.*;
import java.io.*;

public class Sender extends Thread {
    private int port = -1;
    private DataOutputStream outputStream = null;
    Socket socket;
    public Sender(int port)
    {
        this.port = port;
    }

    public void start()
    {

        int count = 0;
        while(count < 10) {
            try {
//                Thread.sleep(32);
                assert this.port != -1;
//                System.out.println("Sender trying to connect to port " + port);
                this.socket = new Socket("localhost", this.port);
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
        synchronized (this) {
            try {
                // Send the message
                this.wait(2);
                outputStream.writeUTF(message);
                outputStream.flush();
//                System.out.println("Sender sent a message on port " + port);
            } catch (IOException e) {
                System.out.println("Error: " + e);
            }
            catch (InterruptedException e) {
                e.printStackTrace();
            }
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

