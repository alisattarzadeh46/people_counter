# centroidtracker.py
# A lightweight centroid-based multi-object tracker.
# Accepts bounding boxes (startX, startY, endX, endY),
# tracks object IDs across frames using centroid proximity.

from collections import OrderedDict
import numpy as np

class CentroidTracker:
    def __init__(self, maxDisappeared: int = 50, maxDistance: float = 50.0):
        """
        Args:
            maxDisappeared: number of consecutive frames an object is
                            allowed to be missing before we deregister it.
            maxDistance: maximum allowed Euclidean distance (in pixels)
                         between existing object and a new detection centroid
                         to be considered the same object.
        """
        # next object ID to assign
        self.nextObjectID = 0
        # object ID -> centroid (x, y)
        self.objects = OrderedDict()
        # object ID -> number of consecutive frames it has been missing
        self.disappeared = OrderedDict()

        self.maxDisappeared = int(maxDisappeared)
        self.maxDistance = float(maxDistance)

    def register(self, centroid):
        """Register a new object with the next available ID."""
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        """Remove an object ID from our tracking dictionaries."""
        if objectID in self.objects:
            del self.objects[objectID]
        if objectID in self.disappeared:
            del self.disappeared[objectID]

    def update(self, rects):
        """
        Update tracked objects with a new list of detection boxes.

        Args:
            rects: list of tuples (startX, startY, endX, endY)

        Returns:
            dict: objectID -> centroid (x, y)
        """
        # If no detections provided, mark existing objects as disappeared
        if len(rects) == 0:
            ids = list(self.disappeared.keys())
            for objectID in ids:
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects

        # Compute the centroid of each input rectangle
        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        # If we have no tracked objects, register all input centroids
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(tuple(inputCentroids[i]))
            return self.objects

        # Grab the set of object IDs and their corresponding centroids
        objectIDs = list(self.objects.keys())
        objectCentroids = list(self.objects.values())

        # Compute distance matrix between each pair (object, input)
        D = self._euclidean_distances(np.array(objectCentroids), inputCentroids)

        # For each object, find the closest input centroid by sorting rows
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        # To keep track of which rows and columns have been examined
        usedRows = set()
        usedCols = set()

        # Perform greedy matching between existing objects and new input centroids
        for (row, col) in zip(rows, cols):
            # If we have already examined either the row or the column, ignore
            if row in usedRows or col in usedCols:
                continue

            # If the distance is too large, ignore this match
            if D[row, col] > self.maxDistance:
                continue

            # Otherwise, update the centroid and reset disappeared counter
            objectID = objectIDs[row]
            self.objects[objectID] = tuple(inputCentroids[col])
            self.disappeared[objectID] = 0

            usedRows.add(row)
            usedCols.add(col)

        # Compute which rows (object IDs) and columns (new detections)
        # were NOT matched
        unusedRows = set(range(0, D.shape[0])).difference(usedRows)
        unusedCols = set(range(0, D.shape[1])).difference(usedCols)

        # Case 1: more existing objects than new detections
        # -> mark unmatched existing objects as disappeared
        if D.shape[0] >= D.shape[1]:
            for row in unusedRows:
                objectID = objectIDs[row]
                self.disappeared[objectID] += 1

                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

        # Case 2: more new detections than existing objects
        # -> register each unmatched new detection as a new object
        else:
            for col in unusedCols:
                self.register(tuple(inputCentroids[col]))

        return self.objects

    @staticmethod
    def _euclidean_distances(a, b):
        """
        Fast pairwise Euclidean distance:
        a: (N, 2), b: (M, 2)
        returns: (N, M) matrix
        """
        # (x1 - x2)^2 + (y1 - y2)^2 = x1^2 + y1^2 + x2^2 + y2^2 - 2*x1*x2 - 2*y1*y2
        a_sq = np.sum(a ** 2, axis=1, keepdims=True)       # (N,1)
        b_sq = np.sum(b ** 2, axis=1, keepdims=True).T     # (1,M)
        ab = a @ b.T                                       # (N,M)
        dists = np.sqrt(np.maximum(a_sq + b_sq - 2.0 * ab, 0.0))
        return dists