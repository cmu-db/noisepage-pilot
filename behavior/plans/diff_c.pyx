import numpy as np

cimport cython
cimport numpy as np

# Import the offsets into the 2D array for relevant positions.

from behavior.plans import (
    LEFT_CHILD_PLAN_NODE_ID_SCHEMA_INDEX,
    PLAN_NODE_ID_SCHEMA_INDEX,
    RIGHT_CHILD_PLAN_NODE_ID_SCHEMA_INDEX,
    TARGET_START_SCHEMA_INDEX,
)

np.import_array()
FTYPE = np.float64
ITYPE = np.int64
ctypedef np.float64_t FTYPE_t
ctypedef np.int64_t ITYPE_t

# Skip checking bounds and wraparound for performance.
@cython.boundscheck(False)
@cython.wraparound(False)
def diff_matrix(np.ndarray[FTYPE_t, ndim=2] matrix):
    cdef int rows = matrix.shape[0]
    cdef int col = matrix.shape[1]

    cdef int plan_id = PLAN_NODE_ID_SCHEMA_INDEX
    cdef int left_id = LEFT_CHILD_PLAN_NODE_ID_SCHEMA_INDEX
    cdef int right_id = RIGHT_CHILD_PLAN_NODE_ID_SCHEMA_INDEX
    cdef int target = TARGET_START_SCHEMA_INDEX

    # Define a 1 dimensional array that has the same length as the number of rows.
    # offset is used to build a mapping offset[i] = Y such that plan node ID [i]
    # corresponds to the data found at matrix[Y].
    cdef np.ndarray[ITYPE_t] offset = np.full(rows, -1, dtype=ITYPE)

    for i in range(rows):
        head = <int>matrix[i][plan_id]
        left = <int>matrix[i][left_id]
        right = <int>matrix[i][right_id]
        if head == -1:
            # The plan_id should never be -1.
            return False

        # If any of the plan node IDs extracted above exceeds the number of rows, then
        # that means this invocation has insufficient data.
        if head >= rows or (left != -1 and left >= rows) or (right != -1 and right >= rows):
            return False

        # If offset[head] has already been populated then that means that this invocation
        # somehow has 2 datapoints with the same plan ID.
        #
        # TODO(wz2): Once we need to difference parallel query execution, revisit this.
        if offset[head] != -1:
            return False
        offset[head] = i

    # access is a BFS order of the query plan tree.
    #
    # We can perfectly size this since [rows] should be how many plans we need to see in
    # order to fully difference the query plan.
    cdef np.ndarray[ITYPE_t] access = np.full(rows, -1, dtype=ITYPE)
    access[0] = 0
    cdef int tail = 1

    for i in range(rows):
        # access[i] describes the plan node ID that we should difference next. It is important
        # that we difference in the proper tree order (parent before child).
        #
        # offset[access[i]] gives us the index into matrix that has the plan node data.
        head = offset[access[i]]
        left = <int>matrix[head][left_id]
        right = <int>matrix[head][right_id]

        if left != -1:
            # In this case, there is a valid left child plan. Difference the metrics and
            # record the left child plan node ID so we difference it next.
            matrix[head, target:col] -= matrix[offset[left], target:col]
            access[tail] = left
            tail += 1

        if right != -1:
            # In this case, there is a valid right child plan. Difference the metrics and
            # record the right child plan node ID so we difference it next.
            matrix[head, target:col] -= matrix[offset[right], target:col]
            access[tail] = right
            tail += 1

    for i in range(rows):
        # Clip all the metrics such that they are non-negative.
        np.clip(matrix[head, target:col], 0, None)

    return True
