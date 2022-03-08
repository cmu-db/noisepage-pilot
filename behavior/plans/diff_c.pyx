import numpy as np

cimport cython
cimport numpy as np

# The following are relevant column offsets for the OU subinvocation matrix used by diff_query_tree().
# These offsets describe the location of a DIFFERENCING_SCHEMA column in the 2D numpy matrix.

from behavior.plans import (
    LEFT_CHILD_PLAN_NODE_ID_SCHEMA_INDEX,
    PLAN_NODE_ID_SCHEMA_INDEX,
    RIGHT_CHILD_PLAN_NODE_ID_SCHEMA_INDEX,
    TARGET_START_SCHEMA_INDEX,
    PlanDiffIncompleteSubinvocationException,
    PlanDiffInvalidDataException,
    PlanDiffUnsupportedParallelException,
)

np.import_array()
FTYPE = np.float64
ITYPE = np.int64
ctypedef np.float64_t FTYPE_t
ctypedef np.int64_t ITYPE_t

# This function accepts a 2D matrix following the DIFFERENCING_SCHEMA with target columns.
# The matrix represents all OUs associated with a given invocation of a particular query template.
# This function differences the OUs based on the plan node structure.
#
# This function does not return any values as it mutates the input 2D matrix in-place.
# This function will raise the following exceptions:
# - PlanDiffIncompleteSubinvocationException when a subinvocation is not complete.
# - PlanDiffInvalidDataException when data is deemed to be invalid.
# - PlanDiffUnsupportedParallelException when a subinvocation contains a parallel OU invocation.
#
# Skip checking bounds and wraparound for performance.
@cython.boundscheck(False)
@cython.wraparound(False)
def diff_query_tree(np.ndarray[FTYPE_t, ndim=2] matrix):
    cdef int rows = np.PyArray_DIMS(matrix)[0]
    # Subtract 1 since the last column is subinvocation_id and we don't "difference" that.
    cdef int col = np.PyArray_DIMS(matrix)[1] - 1

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
            raise PlanDiffInvalidDataException()

        # If any of the plan node IDs extracted above exceeds the number of rows, then
        # that means this invocation has insufficient data.
        if head >= rows or (left != -1 and left >= rows) or (right != -1 and right >= rows):
            raise PlanDiffIncompleteSubinvocationException()

        # If offset[head] has already been populated then that means that this invocation
        # somehow has 2 datapoints with the same plan ID.
        #
        # TODO(wz2): Once we need to difference parallel query execution, revisit this.
        if offset[head] != -1:
            raise PlanDiffUnsupportedParallelException()
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
        # In certain cases, total cost has the property that the child node's total cost might
        # exceed the parent node's total cost (e.g., Limit on top of a SeqScan). In other cases,
        # we may want to adjust NestedLoop()'s total cost as a function of the number of tuples
        # produced by the outer table and the type of inner table probe (e.g., Materialize/IndexScan).
        # TODO(wz2): More robust total cost "differenced" adjustments.
        #
        # We also sanity-clip the rest of the targets to ensure that there are no negative values.
        # This is possible due to variance in data collection.
        matrix[i, target:col] = np.clip(matrix[i, target:col], 0, None)
