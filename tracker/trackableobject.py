# trackableobject.py
# Stores the history of a single tracked object's centroids
# and whether it has already been counted crossing a line.

class TrackableObject:
    def __init__(self, objectID: int, centroid):
        """
        Args:
            objectID: unique ID assigned by the tracker
            centroid: initial (x, y) position
        """
        self.objectID = int(objectID)
        self.centroids = [tuple(centroid)]
        # If the object has already been counted crossing the reference line
        self.counted = False
