import itertools

from plumbum import cli

tables = {
    "warehouse": [
        "w_id",
        "w_ytd",
        "w_tax",
        "w_name",
        "w_street_1",
        "w_street_2",
        "w_city",
        "w_state",
        "w_zip",
    ],
    "item": [
        "i_id",
        "i_name",
        "i_price",
        "i_data",
        "i_im_id",
    ],
    "stock": [
        "s_w_id",
        "s_i_id",
        "s_quantity",
        "s_ytd",
        "s_order_cnt",
        "s_remote_cnt",
        "s_data",
        "s_dist_01",
        "s_dist_02",
        "s_dist_03",
        "s_dist_04",
        "s_dist_05",
        "s_dist_06",
        "s_dist_07",
        "s_dist_08",
        "s_dist_09",
        "s_dist_10",
    ],
    "district": [
        "d_w_id",
        "d_id",
        "d_ytd",
        "d_tax",
        "d_next_o_id",
        "d_name",
        "d_street_1",
        "d_street_2",
        "d_city",
        "d_state",
        "d_zip",
    ],
    "customer": [
        "c_w_id",
        "c_d_id",
        "c_id",
        "c_discount",
        "c_credit",
        "c_last",
        "c_first",
        "c_credit_lim",
        "c_balance",
        "c_ytd_payment",
        "c_payment_cnt",
        "c_delivery_cnt",
        "c_street_1",
        "c_street_2",
        "c_city",
        "c_state",
        "c_zip",
        "c_phone",
        "c_since",
        "c_middle",
        "c_data",
    ],
    "history": [
        "h_c_id",
        "h_c_d_id",
        "h_c_w_id",
        "h_d_id",
        "h_w_id",
        "h_date",
        "h_amount",
        "h_data",
    ],
    "oorder": [
        "o_w_id",
        "o_d_id",
        "o_id",
        "o_c_id",
        "o_carrier_id",
        "o_ol_cnt",
        "o_all_local",
        "o_entry_d",
    ],
    "new_order": [
        "no_w_id",
        "no_d_id",
        "no_o_id",
    ],
    "order_line": [
        "ol_w_id",
        "ol_d_id",
        "ol_o_id",
        "ol_number",
        "ol_i_id",
        "ol_delivery_d",
        "ol_amount",
        "ol_supply_w_id",
        "ol_quantity",
        "ol_dist_info",
    ],
}

tables_filtered = {
    "warehouse": [
        "w_id",
    ],
    "item": [
        "i_id",
    ],
    "stock": [
        "s_w_id",
        "s_i_id",
        "s_quantity",
    ],
    "district": [
        "d_w_id",
        "d_id",
    ],
    "customer": [
        "c_w_id",
        "c_d_id",
        "c_id",
        "c_last",
        "c_first",
    ],
    "oorder": [
        "o_w_id",
        "o_d_id",
        "o_id",
        "o_c_id",
    ],
    "new_order": [
        "no_w_id",
        "no_d_id",
        "no_o_id",
    ],
    "order_line": [
        "ol_w_id",
        "ol_d_id",
        "ol_o_id",
        "ol_i_id",
    ],
}


class GenerateCreateIndexTPCC(cli.Application):
    min_num_cols = cli.SwitchAttr("--min-num-cols", int, mandatory=True)
    max_num_cols = cli.SwitchAttr("--max-num-cols", int, mandatory=True)
    output_sql = cli.SwitchAttr("--output-sql", str, mandatory=True)
    filter_tables = cli.Flag("--filter-tables", default=False)

    def main(self):
        assert 1 <= self.min_num_cols, "Need at least one column."
        assert self.min_num_cols <= self.max_num_cols, "min must be <= max."

        tables_used = tables_filtered if self.filter_tables else tables

        for max_n_cols in range(self.max_num_cols + 1):
            with open(self.output_sql, "w") as f:
                permutations = (
                    (table, permutation)
                    for num_cols in range(self.min_num_cols, max_n_cols + 1)
                    for table, cols in tables_used.items()
                    for permutation in itertools.permutations(cols, num_cols)
                )
                for table, permutation in permutations:
                    cols = ",".join(permutation)
                    cols_name = "_".join(permutation) + "_key"
                    index_name = f"action_{table}_{cols_name}"
                    sql = f"create index if not exists {index_name} on {table} ({cols});"
                    print(sql, file=f)


if __name__ == "__main__":
    GenerateCreateIndexTPCC.run()
