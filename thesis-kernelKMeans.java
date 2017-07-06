/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package waters;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import kmeans.Cluster;
import kmeans.InsufficientMemoryException;
import kmeans.KMeansListener;
import java.util.TreeMap;
import java.util.Map;
import java.util.Collections;
import org.ejml.simple.SimpleMatrix;
import org.ejml.data.*;
import org.ejml.factory.DecompositionFactory;
import org.ejml.interfaces.decomposition.EigenDecomposition;

/**
 * The K-Means algorithm already implemented in the application was
 * a standard version of Lloyd’s method with two modifications. The
 * base of the algorithm was taken from an open-source implementation
 * found at http://www.javaworld.com/article/2076183/build-ci-sdlc/hyper-threaded-java.html. 
 * To turn regular K-Means into Kernel K-Means, two steps were accomplished.
 *
 * First, a variable to store the kernel matrix was added, as well as
 * the method responsible for populating it. The kernel functions implemented
 * include the linear kernel, Gaussian kernel, and the polynomial kernel.
 * Second, the regular distance function used was replaced by the
 * kernel distance function. The three terms of the kernel distance function
 * as described previously are recalculated only when necessary.
 *
 * The first term is a straight lookup in the kernel matrix while the third
 * term is calculated once per cluster per iteration. A member variable of
 * the algorithm’s cluster class was therefore added that tracks the value
 * of this term, and is updated on every reassignment of data points
 * to clusters. The second term has to be calculated on each call of the
 * kernel distance function.
 * 
 * Since regular Kernel K-Means still needs to have the parameter k set,
 * spectral analysis was used to implement an automatic estimation of
 * its optimal value. This approach uses an eigenvalue decomposition of
 * the kernel matrix calculated from the data points. This is convenient,
 * since the matrix is already available.
 *
 * The eigenvalue decomposition is performed by the widely used
 * open-source linear algebra library EJML. The kernel matrix, which
 * is stored as a 2D array of floating-point numbers is converted into a
 * Matrix object of the library and the decomposition function is called.
 * The resulting eigenvalues and corresponding eigenvectors are sorted
 * from the highest eigenvalue and the knee of the plot is
 * determined by the L-method function, which is already available
 * in the K-Means implementation. Similarly to the K-Means function,
 * Kernel K-Means can be used with or without the automatic detection
 * of optimal k.
 *
 * For more: https://is.muni.cz/th/397519/fi_b/
 * @author Branislav Jenco
 */
public class KernelKMeans implements kmeans.KMeans {
    
 // Temporary clusters used during the clustering process.  Converted to
    // an array of the simpler class Cluster at the conclusion.
    private ProtoCluster[] mProtoClusters;

    // Cache of coordinate-to-cluster distances. Number of entries = 
    // number of clusters X number of coordinates.
    private double[][] mDistanceCache;
    
    
    private double[][] mKernelMatrix;

    // Used in makeAssignments() to figure out how many moves are made
    // during each iteration -- the cluster assignment for coordinate n is
    // found in mClusterAssignments[n] where the N coordinates are numbered
    // 0 ... (N-1)
    private int[] mClusterAssignments;

    // 2D array holding the coordinates to be clustered.
    private double[][] mCoordinates;
    // The desired number of clusters and maximum number
    // of iterations.
    private int mK, mMaxIterations;
    // Seed for the random number generator used to select
    // coordinates for the initial cluster centers.
    private long mRandomSeed;
    private float sigma;
    
    // An array of Cluster objects: the output of k-means.
    private Cluster[] mClusters;

    // Listeners to be notified of significant happenings.
    private List<KMeansListener> mListeners = new ArrayList<KMeansListener>(1);
    
    /**
     * Constructor
     * 
     * @param coordinates two-dimensional array containing the coordinates to be clustered.
     * @param k  the number of desired clusters.
     * @param maxIterations the maximum number of clustering iterations.
     * @param randomSeed seed used with the random number generator.
     */
    public KernelKMeans(double[][] coordinates, int k, int maxIterations, 
            long randomSeed, float sigma) {
        mCoordinates = coordinates;
        // Can't have more clusters than coordinates.
        mK = Math.min(k, mCoordinates.length);
        mMaxIterations = maxIterations;
        this.sigma = sigma;
        mRandomSeed = randomSeed;
        mKernelMatrix = new double[mCoordinates.length][mCoordinates.length];
        System.out.println("Kernel K-Means clustering object created with " + this.sigma + " sigma");
    }

    
    private void computeKernelMatrix() {        
        postKMeansMessage("Computing the kernel matrix.");
        System.out.println("Number of points: " + mCoordinates.length);
        for (int i = 0; i < mCoordinates.length; i++) {
            for (int j = 0; j < mCoordinates.length; j++) {
                mKernelMatrix[i][j] = gaussianKernel(mCoordinates[i], mCoordinates[j], sigma);
                // mKernelMatrix[i][j] = polynomialKernel(mCoordinates[i], mCoordinates[j], 3, 7);
                // mKernelMatrix[i][j] = sigmoidKernel(mCoordinates[i], mCoordinates[j], 4);
                // mKernelMatrix[i][j] = linearKernel(mCoordinates[i], mCoordinates[j]); 
            }
        }
        exportKernelMatrix();
        postKMeansMessage("Finished computing the kernel matrix.");
    }

    private double dotProduct(double[] a, double[] b) {
        int len = a.length;
        double dotProduct = 0.0;
        for (int i=0; i<len; i++) {
            dotProduct += a[i] * b[i];
        }
        return dotProduct;
    }
    
    private double linearKernel(double[] a, double[] b) {
        return dotProduct(a,b);
    }

    private double gaussianKernel(double[] a, double[] b, double sigma) {
        int len = a.length;
        double sumSquared = 0.0;
        double v;
        for (int i=0; i<len; i++) {
            v = a[i] - b[i];
            sumSquared += v*v;
        }
        return Math.exp(-sumSquared/(2*sigma*sigma));
        
    }
    
    private double polynomialKernel(double[] a, double[] b, int c, int d) {
        return Math.pow(dotProduct(a,b) + c, d);
        
    }

    private double sigmoidKernel(double[] a, double[] b, double r) {
        return Math.tanh(dotProduct(a,b) + r);
    }
    
    /**
    * Helper function to export and the visualize the kernel matrix for testing purposes
    */
    private void exportKernelMatrix() {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < mCoordinates.length; i++) {
            for (int j = 0; j < mCoordinates.length; j++) {
                sb.append(String.format("%.2f", mKernelMatrix[i][j]));
                if (j < mCoordinates.length - 1) {
                    sb.append(",");
                }
            }
            sb.append(System.getProperty("line.separator"));
        }
        
        try (BufferedWriter br = new BufferedWriter(new FileWriter("matrix.csv"))) {
            br.write(sb.toString());
        } catch(IOException ex) {
            ex.printStackTrace();
        }
    }

    /**
    * converts ejml matrix back into double array 
    */
    public static double[][] EJML2DoubleArray(org.ejml.data.DenseMatrix64F dm) {
        int nr = dm.numRows;
        int nc = dm.numCols;
        double[][] rm = new double[nr][nc];
        for (int r = 0; r < nr; r++)
            for (int c = 0; c < nc; c++)
                rm[r][c] = dm.get(r, c);

        return rm;

    }

    /**
    * Determines the best number of clusters using spectral analysis. Stochastic Simulation of Patterns Using
    * Distance-Based Pattern Modeling, Section 5.3 Alg. 2
    */
    public int determineK() {
        System.out.println("Estimating k...");
        int k = 0;
        int N = mKernelMatrix.length;
        if (mKernelMatrix == null || N == 0) { 
            System.out.println("Kernel matrix is null or empty, can't compute decomposition.");
            return 0;
        }
        System.out.println("Computing eigendecomposition of the kernel matrix...");
        DenseMatrix64F ejmlMat = new DenseMatrix64F(mKernelMatrix);
        EigenDecomposition<DenseMatrix64F> eig = DecompositionFactory.eig(N, true, true);
        eig.decompose(ejmlMat);
        System.out.println("Finished computing eigendecomposition of the kernel matrix.");
        Map<Double, double[]> eigenMap = new TreeMap<Double, double[]>(Collections.reverseOrder());
        for (int i = 0; i < N; i++) {
            eigenMap.put(eig.getEigenvalue(i).getReal(), EJML2DoubleArray(eig.getEigenVector(i))[0]);
        }

        // System.out.println(eigenMap.toString());

        
        double[] plot = new double[N];
        int count = 0;
        for(Map.Entry<Double,double[]> entry : eigenMap.entrySet()) {
            double eigenValue = entry.getKey();
            double[] eigenVector = entry.getValue();
            double eigenDot = 0.0;
            for (int j = 0; j < eigenVector.length; j++) {
                eigenDot += (eigenVector[j]/N)*(eigenVector[j]/N);
            }
            // we don't need to divide each element of the vector by N and then perform the dot product, we can just divide the result by N^2
            plot[count] = eigenValue * eigenDot;

            count++;
        }
        int[] cluster_nums = new int[N];
        for (int i = 0; i < N; ++i) {
            cluster_nums[i] = i+1;
        }
        // export the plot for checking
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < cluster_nums.length; i++) {
            sb.append(plot[i]);
            sb.append(System.getProperty("line.separator"));
        }
        
        try (BufferedWriter br = new BufferedWriter(new FileWriter("lgraph.csv"))) {
            br.write(sb.toString());
        } catch(IOException ex) {
            ex.printStackTrace();
        }

        System.out.println(plot);
        k = LMethod.EstimateClusterNum(cluster_nums, plot);
        

        System.out.println("K is: " + k);

        
        mK = k;
        return k;
    }
    
    /** 
     * Adds a KMeansListener to be notified of significant happenings.
     * 
     * @param l  the listener to be added.
     */
    public void addKMeansListener(KMeansListener l) {
        synchronized (mListeners) {
            if (!mListeners.contains(l)) {  
                mListeners.add(l);
            }
        }
    }
    
    /**
     * Removes a KMeansListener from the listener list.
     * 
     * @param l the listener to be removed.
     */
    public void removeKMeansListener(KMeansListener l) {
        synchronized (mListeners) {
            mListeners.remove(l);
        }
    }
    
    /**
     * Posts a message to registered KMeansListeners.
     * 
     * @param message
     */
    private void postKMeansMessage(String message) {
        if (mListeners.size() > 0) {
            synchronized (mListeners) {
                int sz = mListeners.size();
                for (int i=0; i<sz; i++) {
                    mListeners.get(i).kmeansMessage(message);
                }
            }
        }
    }
    
    /**
     * Notifies registered listeners that k-means is complete.
     * 
     * @param clusters the output of clustering.
     * @param executionTime the number of milliseconds taken to cluster.
     */
    private void postKMeansComplete(Cluster[] clusters, long executionTime) {
        if (mListeners.size() > 0) {
            synchronized (mListeners) {
                int sz = mListeners.size();
                for (int i=0; i<sz; i++) {
                    mListeners.get(i).kmeansComplete(clusters, executionTime);
                }
            }
        }
    }
    
    /**
     * Notifies registered listeners that k-means has failed because of
     * a Throwable caught in the run method.
     * 
     * @param err 
     */
    private void postKMeansError(Throwable err) {
        if (mListeners.size() > 0) {
            synchronized (mListeners) {
                int sz = mListeners.size();
                for (int i=0; i<sz; i++) {
                    mListeners.get(i).kmeansError(err);
                }
            }
        }
    }

    /**
     * Get the clusters computed by the algorithm.  This method should
     * not be called until clustering has completed successfully.
     * 
     * @return an array of Cluster objects.
     */
    public Cluster[] getClusters() {
        return mClusters;
    }
    
    /**
     * Run the clustering algorithm.
     */
    public void run() {

        try {
            
            // Note the start time.
            long startTime = System.currentTimeMillis();
                        
            postKMeansMessage("Kernel K-Means clustering started");
            
            // Compute the kernel matrix
            computeKernelMatrix();
            if (mK == 0) determineK();
            // Randomly initialize the cluster centers creating the
            // array mProtoClusters.
            initCenters();
            
            postKMeansMessage("... centers initialized");

            // Perform the initial computation of distances.
            computeDistances();

            // Make the initial cluster assignments.
            makeAssignments();

            // Number of moves in the iteration and the iteration counter.
            int moves = 0, it = 0;
            
            // Main Loop:
            //
            // Two stopping criteria:
            // - no moves in makeAssignments 
            //   (moves == 0)
            // OR
            // - the maximum number of iterations has been reached
            //   (it == mMaxIterations)
            //
            do {

                // Compute the centers of the clusters that need updating.
                computeCenters();
                
                // Compute the stored distances between the updated clusters and the
                // coordinates.
                computeDistances();

                // Make this iteration's assignments.
                moves = makeAssignments();
                // System.out.println(mProtoClusters[0]);
                it++;
                
                //System.out.println("... iteration " + it + " moves = " + moves);

            } while (moves > 0 && it < mMaxIterations);

            // Transform the array of ProtoClusters to an array
            // of the simpler class Cluster.
            mClusters = generateFinalClusters();
            
            long executionTime = System.currentTimeMillis() - startTime;
            
            postKMeansComplete(mClusters, executionTime);
            
        } catch (Exception ex) {
           
            ex.printStackTrace();
            
        } finally {

            // Clean up temporary data structures used during the algorithm.
            cleanup();

        }
    }

    /**
     * K-means++ initial cluster centers.
     */
    private void initCenters() {

        Random random = new Random(mRandomSeed);
        
        int coordCount = mCoordinates.length;

        // The array mClusterAssignments is used only to keep track of the cluster 
        // membership for each coordinate.  The method makeAssignments() uses it
        // to keep track of the number of moves.
        if (mClusterAssignments == null) {
            mClusterAssignments = new int[coordCount];
            // Initialize to -1 to indicate that they haven't been assigned yet.
            Arrays.fill(mClusterAssignments, -1);
        }
        
        int[] cls_indices = new int[mK];
        final double[] distancSqr=new double[coordCount];
        
        cls_indices[0]=random.nextInt(coordCount);
        
        for (int cluster_found=1;cluster_found<mK;cluster_found++)
        {
            for (int i = 0; i < coordCount; ++i) {
                double d = Double.MAX_VALUE;
                for (int c_i = 0; c_i < cluster_found; ++c_i) {
                    d = Math.min(d, distance(mCoordinates[i], mCoordinates[cls_indices[c_i]]));
                }
                distancSqr[i] = d * d;
            }

            for (int i = 1; i < coordCount; ++i) {
                distancSqr[i] += distancSqr[i-1];
            }

            double random_pick_prob = random.nextDouble() * distancSqr[coordCount - 1];

            int pick_idx = Arrays.binarySearch(distancSqr, random_pick_prob);
            if (pick_idx < 0) {
                pick_idx = -(pick_idx + 1);
            }

            cls_indices[cluster_found] = pick_idx;
        }

        mProtoClusters = new ProtoCluster[mK];
        for (int i=0; i<mK; i++) {
            int coordIndex = cls_indices[i];
            mProtoClusters[i] = new ProtoCluster(mCoordinates[coordIndex], coordIndex);
            System.out.println("Size of cluster " + i + " is " + mProtoClusters[i].mCurrentSize + " and the center has index " + coordIndex);
            mClusterAssignments[cls_indices[i]] = i;
            mProtoClusters[i].updateThirdTerm(mKernelMatrix);
        }
        
        
        
    }
        
    
    private void initCentersOld() {

        Random random = new Random(mRandomSeed);
        
        int coordCount = mCoordinates.length;

        // The array mClusterAssignments is used only to keep track of the cluster 
        // membership for each coordinate.  The method makeAssignments() uses it
        // to keep track of the number of moves.
        if (mClusterAssignments == null) {
            mClusterAssignments = new int[coordCount];
            // Initialize to -1 to indicate that they haven't been assigned yet.
            Arrays.fill(mClusterAssignments, -1);
        }

        // Place the coordinate indices into an array and shuffle it.
        int[] indices = new int[coordCount];
        for (int i = 0; i < coordCount; i++) {
            indices[i] = i;
        }
        for (int i = 0, m = coordCount; m > 0; i++, m--) {
            int j = i + random.nextInt(m);
            if (i != j) {
                // Swap the indices.
                indices[i] ^= indices[j];
                indices[j] ^= indices[i];
                indices[i] ^= indices[j];
            }
        }

        mProtoClusters = new ProtoCluster[mK];
        for (int i=0; i<mK; i++) {
            int coordIndex = indices[i];
            mProtoClusters[i] = new ProtoCluster(mCoordinates[coordIndex], coordIndex);
            mProtoClusters[i].updateThirdTerm(mKernelMatrix);
            mClusterAssignments[indices[i]] = i;
        }
    }

    /**
     * Recompute the centers of the protoclusters with 
     * update flags set to true.
     */
    private void computeCenters() {
        
        int numClusters = mProtoClusters.length;
        
        // Sets the update flags of the protoclusters that haven't been deleted and
        // whose memberships have changed in the iteration just completed.
        //
        for (int c = 0; c < numClusters; c++) {
            ProtoCluster cluster = mProtoClusters[c];
            //System.out.println("Cluster " + c + " has size " + cluster.mCurrentSize);
            if (cluster.getConsiderForAssignment()) {
                if (!cluster.isEmpty()) {
                    // This sets the protocluster's update flag to
                    // true only if its membership changed in last call
                    // to makeAssignments().  
                    cluster.setUpdateFlag();
                    // If the update flag was set, update the center.
                    if (cluster.needsUpdate()) {
                        cluster.updateCenter(mCoordinates);
                        cluster.updateThirdTerm(mKernelMatrix);
                    }
                } else {
                    // When a cluster loses all of its members, it
                    // falls out of contention.  So it is possible for
                    // k-means to return fewer than k clusters.
                    cluster.setConsiderForAssignment(false);
                }
            }
        }
    }

    /** 
     * Compute distances between coodinates and cluster centers,
     * storing them in the distance cache.  Only distances that
     * need to be computed are computed.  This is determined by
     * distance update flags in the protocluster objects.
     */
    private void computeDistances() throws InsufficientMemoryException {
        
        int numCoords = mCoordinates.length;
        int numClusters = mProtoClusters.length;

        if (mDistanceCache == null) {
            // Explicit garbage collection to reduce likelihood of insufficient
            // memory.
            System.gc();
            // Ensure there is enough memory available for the distances.  
            // Throw an exception if not.
            long memRequired = 8L * numCoords * numClusters;
            if (Runtime.getRuntime().freeMemory() < memRequired) {
                throw new InsufficientMemoryException();
            }
            // Instantiate an array to hold the distances between coordinates
            // and cluster centers
            mDistanceCache = new double[numCoords][numClusters];
        }

        for (int coord=0; coord < numCoords; coord++) {
            // Update the distances between the coordinate and all
            // clusters currently in contention with update flags set.
            for (int clust=0; clust<numClusters; clust++) {
                ProtoCluster cluster = mProtoClusters[clust];
                if (cluster.getConsiderForAssignment() && cluster.needsUpdate()) {
                    mDistanceCache[coord][clust] = 
                        kernelDistance(mCoordinates[coord], coord, mKernelMatrix, cluster);
                        //distance(mCoordinates[coord], cluster.getCenter());
                }
            }
        }
        
    }
    
    /** 
     * Assign each coordinate to the nearest cluster.  Called once
     * per iteration.  Returns the number of coordinates that have
     * changed their cluster membership.
     */
    private int makeAssignments() {

        int moves = 0;
        int coordCount = mCoordinates.length;

        // Checkpoint the clusters, so we'll be able to tell
        // which ones have changed after all the assignments have been
        // made.
        int numClusters = mProtoClusters.length;
        for (int c = 0; c < numClusters; c++) {
            if (mProtoClusters[c].getConsiderForAssignment()) {
                mProtoClusters[c].checkPoint();
            }
        }

        // Now do the assignments.
        for (int i = 0; i < coordCount; i++) {
            int c = nearestCluster(i);
            mProtoClusters[c].add(i);
            if (mClusterAssignments[i] != c) {
                mClusterAssignments[i] = c;
                moves++;
            }
        }

        return moves;
    }

    /**
     * Find the nearest cluster to the coordinate identified by
     * the specified index.
     */
    private int nearestCluster(int ndx) {
        int nearest = -1;
        double min = Double.MAX_VALUE;
        int numClusters = mProtoClusters.length;
        for (int c = 0; c < numClusters; c++) {
            if (mProtoClusters[c].getConsiderForAssignment()) {
                double d = mDistanceCache[ndx][c];
                if (d < min) {
                    min = d;
                    nearest = c;
                }
            }
        }
        return nearest;
    }
 
    /**
     * Compute the euclidean distance between the two arguments.
     */
    private static double distance(double[] coord, double[] center) {
        int len = coord.length;
        double sumSquared = 0.0;
        for (int i=0; i<len; i++) {
            double v = coord[i] - center[i];
            sumSquared += v*v;
        }
        return Math.sqrt(sumSquared);
    }
     
    /**
     * Compute the distance between a point and a cluster centroid in transformed feature space.
     */
    private static double kernelDistance(double[] coord, int coord_idx, double[][] kernel, ProtoCluster cluster) {
        int N = cluster.mCurrentSize;
        double first_term = kernel[coord_idx][coord_idx];
        double second_term = 0.0;
        double third_term = cluster.getThirdTerm();
        for (int i = 0; i < N; i++) {
            second_term += kernel[cluster.getMembership()[i]][coord_idx];
        }
        second_term /= (double) N;
        double res = Math.sqrt(first_term - 2*second_term + third_term);
        return res;
        
            
    }

    // private void testDistance() {
    //     //double[] x = [3.0, -2.0, 4.0];
    //     // Coordinates
    //     double[][] coordinates = new double[][]{new double[] {3.0, -2.0, 4.0}, new double[]{1.0,1.0,1.0}, new double[] {1.0,2.0,1.0},new double[] {2.0,1.0,1.0},new double[] {2.0,2.0,1.0}, new double[] {2.0,2.0,2.0}};
    //     // Make matrix
    //     double[][] K = new double[coordinates.length][coordinates.length];
    //     for (int i = 0; i < coordinates.length; i++) {
    //         for (int j = 0; j < coordinates.length; j++) {
    //             //mKernelMatrix[i][j] = gaussianKernel(mCoordinates[i], mCoordinates[j], mCoordinates.length/2);
    //             //mKernelMatrix[i][j] = polynomialKernel(mCoordinates[i], mCoordinates[j], 0, 2);
    //             K[i][j] = linearKernel(coordinates[i], coordinates[j]); 
    //         }
    //     }
    //     System.out.println("Matrix: " + K.toString());

    //     // Make cluster
    //     ProtoCluster cluster = new ProtoCluster(coordinates[1], 1);
    //     for (int i = 2; i < 6; i++) {
    //         cluster.add(i);
    //     }
    //     cluster.updateCenter(coordinates);
    //     cluster.updateThirdTerm(K);

    //     System.out.println("D: " + distance(coordinates[0], cluster.getCenter()));
    //     System.out.println("KD: " + kernelDistance(coordinates[0], 0, K, cluster));

    // }


    /**
     * Generate an array of Cluster objects from mProtoClusters.
     * 
     * @return array of Cluster object references.
     */
    private Cluster[] generateFinalClusters() {
        
        int numClusters = mProtoClusters.length;
        System.out.println("Number of protoclusters when we generate finals is " + numClusters);
        
        // Convert the proto-clusters to the final Clusters.
        //
        // - accumulate in a list.
        List<Cluster> clusterList = new ArrayList<Cluster>(numClusters);
        for (int c = 0; c < numClusters; c++) {
            ProtoCluster pcluster = mProtoClusters[c];
            if (!pcluster.isEmpty()) {
                Cluster cluster = new Cluster(pcluster.getMembership(), pcluster.getCenter());
                clusterList.add(cluster);
            } else {
                System.out.println("cluster " + c + " is empty");
            }
        }
    
        // - convert list to an array.
        Cluster[] clusters = new Cluster[clusterList.size()];
        clusterList.toArray(clusters);

        return clusters;
    }
    
    /**
     * Clean up items used by the clustering algorithm that are no longer needed.
     */
    private void cleanup() {
        mProtoClusters = null;
        mDistanceCache = null;
        mClusterAssignments = null;
        mKernelMatrix = null;
    }

    /**
     * Cluster class used temporarily during clustering.  Upon completion,
     * the array of ProtoClusters is transformed into an array of
     * Clusters.
     */
    private class ProtoCluster {

        // The previous iteration's cluster membership and
        // the current iteration's membership.  Compared to see if the
        // cluster has changed during the last iteration.
        private int[] mPreviousMembership;
        private int[] mCurrentMembership;
        private int mCurrentSize;

        // The cluster center.
        private double[] mCenter;
        
        // The third term in the kernel k means calculation, it is constant for 
        // one cluster in an iteration
        private double third_term = 0.0;

        // Born true, so the first call to updateDistances() will set all the
        // distances.
        private boolean mUpdateFlag = true;
        // Whether or not this cluster takes part in the operations.
        private boolean mConsiderForAssignment = true;

        /**
         * Constructor
         * 
         * @param center  the initial cluster center.
         * @param coordIndex  the initial member. 
         */
        ProtoCluster(double[] center, int coordIndex) {
            mCenter = (double[]) center.clone();
            // No previous membership.
            mPreviousMembership = new int[0];
            // Provide space for 10 members to be added initially.
            mCurrentMembership = new int[10];
            mCurrentSize = 0;
            add(coordIndex);
        }
        
        
        /**
         * Get the members of this protocluster.
         * 
         * @return an array of coordinate indices.
         */
        int[] getMembership() {
            trimCurrentMembership();
            return mCurrentMembership;
        }
        
        /**
         * Get the protocluster's center.
         * 
         * @return
         */
        double[] getCenter() {
            return mCenter;
        }
        
        /**
         * Reduces the length of the array of current members to
         * the number of members.
         */
        void trimCurrentMembership() {
            if (mCurrentMembership.length > mCurrentSize) {
                int[] temp = new int[mCurrentSize];
                System.arraycopy(mCurrentMembership, 0, temp, 0, mCurrentSize);
                mCurrentMembership = temp;
            }
        }

        /**
         * Add a coordinate to the protocluster.
         * 
         * @param ndx index of the coordinate to be added.
         */
        void add(int ndx) {
            // Ensure there's space to add the new member.
            if (mCurrentSize == mCurrentMembership.length) {
                // If not, double the size of mCurrentMembership.
                int newCapacity = Math.max(10, 2*mCurrentMembership.length);
                int[] temp = new int[newCapacity];
                System.arraycopy(mCurrentMembership, 0, temp, 0, mCurrentSize);
                mCurrentMembership = temp;
            }
            // Add the index.
            mCurrentMembership[mCurrentSize++] = ndx;
        }

        /**
         * Does the protocluster contain any members?
         * 
         * @return true if the cluster is empty.
         */
        boolean isEmpty() {
            return mCurrentSize == 0;
        }

        /**
         * Compares the previous and the current membership.
         * Sets the update flag to true if the membership
         * changed in the previous call to makeAssignments().
         */
        void setUpdateFlag() {
            // Trim the current membership array length down to the
            // number of members.
            trimCurrentMembership();
            mUpdateFlag = false;
            if (mPreviousMembership.length == mCurrentSize) {
                for (int i=0; i<mCurrentSize; i++) {
                    if (mPreviousMembership[i] != mCurrentMembership[i]) {
                        mUpdateFlag = true;
                        break;
                    }
                }
            } else { // Number of members has changed.
                mUpdateFlag = true;
            }
        }

        /**
         * Clears the current membership after copying it to the
         * previous membership.
         */
        void checkPoint() {
            mPreviousMembership = mCurrentMembership;
            mCurrentMembership = new int[10];
            mCurrentSize = 0;
        }

        /**
         * Is this protocluster currently in contention?
         * 
         * @return true if this cluster is still in the running.
         */
        boolean getConsiderForAssignment() {
            return mConsiderForAssignment;
        }

        /**
         * Set the flag to indicate that this protocluster is
         * in or out of contention.
         * 
         * @param b
         */
        void setConsiderForAssignment(boolean b) {
            mConsiderForAssignment = b;
        }

        /**
         * Get the value of the update flag.  This value is
         * used to determine whether to update the cluster center and
         * whether to recompute distances to the cluster.
         * 
         * @return the value of the update flag.
         */
        boolean needsUpdate() {
            return mUpdateFlag;
        }

        /**
         * Update the cluster center.
         * 
         * @param coordinates the array of coordinates.
         */
        void updateCenter(double[][] coordinates) {
            Arrays.fill(mCenter, 0.0);
            if (mCurrentSize > 0) {
                for (int i=0; i<mCurrentSize; i++) {
                    double[] coord = coordinates[mCurrentMembership[i]];
                    for (int j=0; j<coord.length; j++) {
                        mCenter[j] += coord[j];
                    }
                }
                for (int i=0; i<mCenter.length; i++) {
                    mCenter[i] /= mCurrentSize;
                }
            }
        }

        double updateThirdTerm(double[][] kernelMatrix) {
            for (int i = 0; i < mCurrentSize; i++) {
                for (int j = 0; j < mCurrentSize; j++) {
                    third_term += kernelMatrix[mCurrentMembership[i]][mCurrentMembership[j]];
                }
            }
            third_term /= (mCurrentSize*mCurrentSize);
            return third_term;
        }

        double getThirdTerm() {
          return third_term;
        }

        @Override
        public String toString() {
            return Arrays.toString(mCurrentMembership);
        }
    }

    
}
