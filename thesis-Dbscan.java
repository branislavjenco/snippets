package waters;

import java.util.ArrayList;
import java.util.List;
import javax.vecmath.Vector3f;


/**
 * The functionality of the Dbscan class is similar to the previous classes,
 * with the difference that the algorithm was implemented from scratch.
 * After instantiating the class, the member variables MinPts and Eps are
 * set according to user input and the data points are transformed into
 * Point objects, an inner class of Dbscan which keeps track of currently
 * assigned cluster index for the point. A cluster index of 0 marks a point
 * as noise (if there is any present) and an index of -1 marks the point as
 * yet unclassified (as per the algorithm, all points are set to unclassified
 * at the start).
 *
 * @author Branislav Jenco
 */
public class Dbscan {
    
    private final double eps;
    private final int minPts;
    private final ArrayList<Point> points;
    private boolean noisePresent = false;
    private int clusterNum;
    
    public Dbscan(ArrayList<Vector3f> points, double eps, int minPts) {
        this.eps = eps;
        this.minPts = minPts;
        this.points = new ArrayList<>();
        for (int i = 0; i < points.size(); i++) {
            this.points.add(new Point(points.get(i), i));
        }
    }
    
    public InletCluster[] exportClusters() {
        ArrayList<ArrayList> indices = new ArrayList<>();
        if (noisePresent) clusterNum = clusterNum + 1;

        for (int i = 0; i < clusterNum; i++) {
            indices.add(new ArrayList<>());
        }

        if (noisePresent) {
            System.out.println("Noise present.");
            for (Point point : points) {
                indices.get(point.clusterId).add(point.index);
            }
        } else {
            for (Point point : points) {
                indices.get(point.clusterId - 1).add(point.index);
            }
        }
       
        InletCluster clusters[] = new InletCluster[clusterNum];
        for (int i = 0; i < clusterNum; i++) {
            clusters[i] = new InletCluster(indices.get(i));
        }
        return clusters;
    }
    
    /**
    * Main clustering method. For more, see https://www.aaai.org/Papers/KDD/1996/KDD96-037.pdf
    */
    public void cluster() {
        int C = 0;
        for (Point p : points) {
            if (p.visited) {
                continue;
            } 
            p.visited = true;
            ArrayList<Point> neighbors = regionQuery(p);
            if (neighbors.size() < minPts) {
                p.clusterId = 0;
                noisePresent = true;
            } else {
                C = C + 1;
                p.clusterId = C;
                expandCluster(p, C, neighbors);
            }
        }
        clusterNum = C;
    }

    private void expandCluster(Point p, int C, ArrayList<Point> neighbors) {
                int count = 0;
        while(count < neighbors.size()) {
            Point pprime = neighbors.get(count);
            if (pprime.clusterId == -1) pprime.clusterId = C;
            if (!pprime.visited) {
                pprime.visited = true;
                ArrayList<Point> newNeighbors = regionQuery(pprime);
                if (newNeighbors.size() >= minPts) neighbors.addAll(newNeighbors);
            }
            count++;
        } 
    } 
    
    private ArrayList<Point> regionQuery(Point p) {
        ArrayList<Point> result = new ArrayList<>();
        for (Point point : points) {
            if (distance(p.position, point.position) < eps) {
                result.add(point);
            }
        }
        return result;
    }
    
    /**
     * Compute the 3D euclidean distance between the two points p and q.
     */
    private static double distance(Vector3f p, Vector3f q) {
        double sumSquared = (p.x - q.x)*(p.x - q.x) + (p.y - q.y)*(p.y - q.y) + (p.z - q.z)*(p.z - q.z);
        return Math.sqrt(sumSquared);
    }
    
    private class Point {
        // 0 for noise
        int clusterId;
        int index;
        boolean visited;
       
        Vector3f position;

        public Point(Vector3f position, int index) {
            this.clusterId = -1;
            this.visited = false;
            this.position = position;
            this.index = index;
            
        }
        
        @Override
        public String toString() {
            return "P" + this.index + "c" + this.clusterId;
        }
        
    }
    
    private class DBcluster {
        ArrayList<Point> memberPoints;
        public DBcluster(){
            
        }

        public ArrayList<Point> getMemberPoints() {
            return memberPoints;
        }
        public void addPoint(Point p) {
            memberPoints.add(p);
        }
        
    }
    
    
}
