from connector import Connector
import logparsing_copy

class Workload:
    def __init__(self, workload_csv_file_name: str, db_connector: Connector):

        self.file_name = workload_csv_file_name
        self.conn = db_connector

        # TODO: cleaner format to store these?
        self._parse()


    def _parse(self):
        # execute log parsing for workload, store each type of col_refs in a per-table basis
        parsed = logparsing_copy.parse_csv_log(self.file_name)
        filtered = logparsing_copy.aggregate_templates(parsed, self.conn)
        self.colrefs = logparsing_copy.get_workload_colrefs(filtered)

    def get_where_colrefs(self):
        return self.colrefs["where_colrefs"]

    def get_group_by_colrefs(self):
        return self.colrefs["group_by_colrefs"]


if __name__ == "__main__":
    wkld = Workload('logs/epinions.csv', Connector())
    print("where_colrefs", wkld.get_group_by_colrefs())
    print("group_by_colrefs", wkld.get_where_colrefs())
    # looks like:
    # where_colrefs defaultdict(<function get_workload_colrefs.<locals>.<dictcomp>.<lambda> at 0x7f195300eaf0>, {'review': defaultdict(<class 'numpy.uint64'>, {('u_id',): 1.0})})
    # group_by_colrefs defaultdict(<function get_workload_colrefs.<locals>.<dictcomp>.<lambda> at 0x7f1965f67040>, {'trust': defaultdict(<class 'numpy.uint64'>, {('target_u_id', 'source_u_id'): 88.0, ('source_u_id',): 37.0}), 'review': defaultdict(<class 'numpy.uint64'>, {('i_id',): 99.0, ('u_id',): 30.0, ('i_id', 'u_id'): 30.0, ('u_id', 'i_id'): 28.0}), 'item': defaultdict(<class 'numpy.uint64'>, {('i_id',): 62.0}), 'useracct': defaultdict(<class 'numpy.uint64'>, {('u_id',): 61.0})})

