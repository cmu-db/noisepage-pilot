import datetime
import math
import random

import psycopg
from plumbum import cli, local

NOOP_ACTION = "SELECT 1;"


class IndexPickerCLI(cli.Application):
    database_game_path = cli.SwitchAttr("--database-game-path", str, mandatory=True)
    batch_size = cli.SwitchAttr("--batch-size", int, default=100)
    tmp_actions_path = cli.SwitchAttr(
        "--tmp-actions-path", str, default="/tmp/actions.csv"
    )

    def main(self, *args):
        # Preprocess the arguments to reuse some of them.
        args = list(args)
        # Swap out the actions path to do action batching.
        actions_path_idx = args.index("--actions_path") + 1
        actions_path = args[actions_path_idx]
        args[actions_path_idx] = self.tmp_actions_path
        # Extract the database connection string to apply recommended actions.
        db_conn_string = args[args.index("--db_conn_string") + 1]
        args = tuple(args)

        # Read the original actions path.
        with open(actions_path, "r", encoding="utf-8") as all_actions:
            actions = all_actions.readlines()
            actions = [sql.strip() for sql in actions]

        # Bind the commands for the database_game binary invocation.
        db = local[self.database_game_path][args]

        # Seed the RNG, here is as good a place as any.
        random.seed(15799)

        def action_batches():
            """
            Chunk the list of actions into batches.

            Actions are chunked into manageable batches to improve the
            runtime of the CFR action recommendation algorithm.
            As the batch size approaches "all of the actions",
            action recommendation approaches optimality.

            However, by having a batch size, recommendation is effectively
            greedy in the input list. For example, consider the extreme
            case where batch size is set to one.

            TODO(WAN): A better approach?

            Returns
            -------
            next_batch : Iterable[str]
                The next chunk of actions.
            """
            for idx in range(0, len(actions), self.batch_size):
                next_batch = [NOOP_ACTION]
                max_idx = min(len(actions), idx + self.batch_size)
                next_batch.extend(actions[idx:max_idx])
                yield next_batch

        def write_action_batch(batch):
            """
            Write the current batch of actions to a temporary CSV file.

            TODO(WAN):
                This exists because currently, the input interface to the
                action selection binary is a .CSV file of actions.
            """
            with open(self.tmp_actions_path, "w", encoding="utf-8") as actions_file:
                for action in batch:
                    print(action.strip(), file=actions_file)

        actions_taken = []
        previous_returns = -math.inf

        # Loop through all of the action batches, applying recommended
        # actions as long as there is an improvement in overall returns.
        with psycopg.connect(db_conn_string) as conn:
            with conn.cursor() as cursor:
                for batch in action_batches():
                    batch = list(batch)

                    # The current behavior of action recommendation is to
                    # consider all the items of a batch together.
                    # Therefore the following shuffle should have no effect
                    # on recommendations; this is done for robustness.
                    random.shuffle(batch)

                    # Prepare input to action selection; see docstring.
                    write_action_batch(batch)

                    # Loop action recommendation for the same batch until
                    # there are no more good actions in the batch.
                    while True:
                        # Pick the next action.
                        (retcode, stdout, stderr) = db.run()
                        assert retcode == 0, f"Got return code: {retcode}"

                        # Get the recommended action.
                        # TODO(WAN): Hack. Better interface?
                        action = stdout.strip()
                        for line in stderr.split("\n"):
                            if line.startswith("\tFinal returns:"):
                                val_str = line.split(":")[1].strip()
                                current_returns = float(val_str)

                        # Always remove the recommended action from the
                        # current batch of actions.
                        if action in batch:
                            batch.remove(action)

                        # Despite the above removal, an action may still
                        # be recommended multiple times. This is because
                        # actions.csv is not rewritten on each update;
                        # the cost of rewriting is only paid when repeats
                        # start showing up.
                        if action in actions_taken:
                            # A repeat showed up. Rewrite input and retry.
                            write_action_batch(batch)
                            continue

                        # Because action recommendation is on batched input,
                        # it can find optimal actions within a batch,
                        # but it cannot find optimal actions across batches.
                        # Therefore recommended actions are only applied if
                        # they improve the overall returns.
                        # TODO(WAN): heuristic for improvement, say, 10%?
                        if action == NOOP_ACTION or current_returns <= previous_returns:
                            # The recommended action is not an improvement.
                            # Start a new batch.
                            break
                        # The recommended action is an improvement.
                        # Apply the recommended action.
                        assert action not in actions_taken
                        cursor.execute(action)
                        conn.commit()
                        # Log the action taken.
                        actions_taken.append(action)
                        print(
                            f"{datetime.datetime.now()} "
                            f"Applied action due to improved returns "
                            f"(previous {previous_returns} -> "
                            f"current {current_returns}): {action}",
                            flush=True,
                        )
                        previous_returns = current_returns


if __name__ == "__main__":
    IndexPickerCLI.run()
