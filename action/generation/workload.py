from connector import Connector
from logparsing_copy import *


class Workload:
    def __init__(self, workload_csv_file_name: str, db_connector: Connector):

        self.file_name = workload_csv_file_name
        self.conn = db_connector

        # TODO: cleaner format to store these?
        self.parsed_group_by_colrefs = set()
        self.parsed_where_colrefs = set()

    def parse():
        # execute log parsing for workload
        parsed = logparsing_copy.parse_csv_log(self.file_name)
        filtered = logparsing_copy.aggregate_templates(parsed, self.conn)
        colrefs = logparsing_copy.get_workload_colrefs(filtered)
