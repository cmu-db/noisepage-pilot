#include <string>

#include "database.h"
#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"
#include "open_spiel/abseil-cpp/absl/random/discrete_distribution.h"
#include "open_spiel/algorithms/cfr.h"
#include "open_spiel/algorithms/mcts.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

// General flags.
ABSL_FLAG(std::string, game_name, "database_game", "The game to solve.");
ABSL_FLAG(std::string, db_conn_string,
          "host=127.0.0.1 port=5432 dbname=spiel user=spiel password=spiel sslmode=disable application_name=psql",
          "Database connection string.");
ABSL_FLAG(std::string, forecast_path, "./forecast.csv", "Path to CSV of forecasted SQL queries.");
ABSL_FLAG(std::string, actions_path, "./actions.csv", "Path to CSV of possible SQL actions.");
ABSL_FLAG(int, max_tuning_actions, 1, "Maximum number of tuning actions before the game ends.");
ABSL_FLAG(bool, use_hypopg, true, "True if hypopg should be used for faking index builds.");
ABSL_FLAG(bool, use_microservice, false, "True if microservice should be used for inference.");

// Solver type.
ABSL_FLAG(std::string, solver_type, "cfr", "Solver to use. {cfr,mcts}.");

// Solver: CFR.
ABSL_FLAG(int, cfr_num_iters, 1, "CFR: How many iterations to run for.");
ABSL_FLAG(int, cfr_simulate_every, 1, "CFR: How often to simulate the game with the current policy.");
ABSL_FLAG(std::string, cfr_simulation_policy, "current", "CFR: Which policy to use in simulation. {average,current}.");
ABSL_FLAG(std::string, cfr_simulation_policy_play, "max", "CFR: How to use the policy. {max,weighted}.");
ABSL_FLAG(uint_fast32_t, cfr_simulation_seed, 15721,
          "CFR: Seed for all simulation RNG, e.g., selection from weighted policy.");

// Solver: MCTS.
ABSL_FLAG(int, mcts_rollout_count, 10, "MCTS: How many rollouts per evaluation.");
ABSL_FLAG(uint_fast32_t, mcts_seed, 15721, "MCTS: Seed.");
ABSL_FLAG(double, mcts_uct_c, 2, "MCTS: UCT exploration constant.");
ABSL_FLAG(int, mcts_max_simulations, 10000, "MCTS: How many simulations to run.");
ABSL_FLAG(int, mcts_max_memory_mb, 1000, "MCTS: Maximum memory used before stopping search.");
ABSL_FLAG(bool, mcts_solve, true, "MCTS: True to use MCTS-Solver which backs up solved states.");
ABSL_FLAG(bool, mcts_verbose, false, "MCTS: True to show MCTS stats of possible moves.");

std::unique_ptr<open_spiel::Bot> CreateMCTSBot(const open_spiel::Game &game) {
  auto evaluator = std::make_shared<open_spiel::algorithms::RandomRolloutEvaluator>(
      absl::GetFlag(FLAGS_mcts_rollout_count), absl::GetFlag(FLAGS_mcts_seed));
  return std::make_unique<open_spiel::algorithms::MCTSBot>(
      game, std::move(evaluator), absl::GetFlag(FLAGS_mcts_uct_c), absl::GetFlag(FLAGS_mcts_max_simulations),
      absl::GetFlag(FLAGS_mcts_max_memory_mb), absl::GetFlag(FLAGS_mcts_solve), absl::GetFlag(FLAGS_mcts_seed),
      absl::GetFlag(FLAGS_mcts_verbose));
}

int main(int argc, char **argv) {
  absl::ParseCommandLine(argc, argv);

  // Parse the game's parameters.
  open_spiel::GameParameters params;
  params.emplace("db_conn_string", absl::GetFlag(FLAGS_db_conn_string));
  params.emplace("forecast_path", absl::GetFlag(FLAGS_forecast_path));
  params.emplace("actions_path", absl::GetFlag(FLAGS_actions_path));
  params.emplace("max_tuning_actions", absl::GetFlag(FLAGS_max_tuning_actions));
  params.emplace("use_hypopg", absl::GetFlag(FLAGS_use_hypopg));
  params.emplace("use_microservice", absl::GetFlag(FLAGS_use_microservice));

  std::cerr << "Generic game parameters:" << std::endl;
  for (const auto &param : params) {
    std::cerr << "\t" << param.first << ": " << param.second.ToString() << std::endl;
  }
  std::cerr << std::endl;

  // Determine which solver to use.
  std::string solver_type = absl::GetFlag(FLAGS_solver_type);

  // Load the game.
  std::shared_ptr<const open_spiel::Game> game = open_spiel::LoadGame(absl::GetFlag(FLAGS_game_name), params);

  if (solver_type == "cfr") {
    std::cerr << "Solver: CFR" << std::endl;

    int num_iters = absl::GetFlag(FLAGS_cfr_num_iters);
    int simulate_every = absl::GetFlag(FLAGS_cfr_simulate_every);
    std::string simulation_policy = absl::GetFlag(FLAGS_cfr_simulation_policy);
    std::string simulation_policy_play = absl::GetFlag(FLAGS_cfr_simulation_policy_play);
    uint32_t simulation_seed = absl::GetFlag(FLAGS_cfr_simulation_seed);

    std::cerr << "\tCFR game parameters: " << std::endl;
    std::cerr << "\t\tcfr_num_iters: " << num_iters << std::endl;
    std::cerr << "\t\tcfr_simulate_every: " << simulate_every << std::endl;
    std::cerr << "\t\tcfr_simulation_policy: " << simulation_policy << std::endl;
    std::cerr << "\t\tcfr_simulation_policy_play: " << simulation_policy_play << std::endl;
    std::cerr << "\t\tcfr_simulation_seed: " << simulation_seed << std::endl;
    std::cerr << std::endl;

    open_spiel::algorithms::CFRSolver solver(*game);
    std::mt19937 rng(simulation_seed);

    for (int iteration = 0; iteration < num_iters; ++iteration) {
      solver.EvaluateAndUpdatePolicy();

      bool should_simulate = iteration % simulate_every == 0;
      bool last_iteration = iteration == num_iters - 1;
      if (should_simulate || last_iteration) {
        std::cerr << "[Policy/" << simulation_policy << ",Play/" << simulation_policy_play << "] ";
        std::cerr << "Iteration " << iteration << std::endl;

        // Get the relevant policy from the Solver.
        std::shared_ptr<open_spiel::Policy> policy;
        {
          if (simulation_policy == "current") {
            policy = solver.CurrentPolicy();
          } else if (simulation_policy == "average") {
            policy = solver.AveragePolicy();
          } else {
            throw std::runtime_error(absl::StrCat("Invalid cfr_simulation_policy: ", simulation_policy));
          }
        }

        // Simulate an entire iteration of the game.
        std::unique_ptr<open_spiel::State> state = game->NewInitialState();
        while (!state->IsTerminal()) {
          const open_spiel::ActionsAndProbs &actions_and_probabilities = policy->GetStatePolicy(*state);
          // Pick an action according to the policy.
          open_spiel::Action action;
          {
            std::vector<double> distribution;
            for (const auto &action_and_probability : actions_and_probabilities) {
              distribution.emplace_back(action_and_probability.second);
            }

            if (simulation_policy_play == "max") {
              // Pick the action with the highest probability.
              auto it = max_element(actions_and_probabilities.begin(), actions_and_probabilities.end(),
                                    [](const auto &lhs, const auto &rhs) { return lhs.second < rhs.second; });
              action = it->first;
            } else if (simulation_policy_play == "weighted") {
              // Pick an action randomly, weighted by the policy.
              absl::discrete_distribution<> dis(distribution.begin(), distribution.end());
              action = state->LegalActions()[dis(rng)];
            } else {
              throw std::runtime_error(absl::StrCat("Invalid cfr_simulation_policy_play: ", simulation_policy_play));
            }

            std::cerr << "\tDistribution: " << absl::StrJoin(distribution, ",") << std::endl;
            auto db_state = dynamic_cast<open_spiel::database::DatabaseState *>(state.get());
            std::cerr << "\tPicked action: " << action << " " << db_state->GetActionSQL(action) << std::endl;
          }

          // Apply the picked action.
          state->ApplyAction(action);
        }

        // The game has terminated. Output the cost incurred.
        const auto &returns = state->Returns();
        std::cerr << "\tFinal returns: " << returns[0] << std::endl;
      }
    }

    std::unique_ptr<open_spiel::State> state = game->NewInitialState();
    const open_spiel::ActionsAndProbs &actions_and_probabilities = solver.CurrentPolicy()->GetStatePolicy(*state);
    auto it = max_element(actions_and_probabilities.begin(), actions_and_probabilities.end(),
                          [](const auto &lhs, const auto &rhs) { return lhs.second < rhs.second; });
    open_spiel::Action action = it->first;
    auto db_state = dynamic_cast<open_spiel::database::DatabaseState *>(state.get());
    std::cout << db_state->GetActionSQL(action) << std::endl;
  } else if (solver_type == "mcts") {
    std::cerr << "Solver: MCTS" << std::endl;
    throw std::runtime_error("Hook up the bot.");
    //  auto mcts_bot = CreateMCTSBot(*game);
    //  open_spiel::Action mcts_action = mcts_bot->Step(*state);
    //  state->ApplyAction(mcts_action);
  } else {
    throw std::runtime_error(absl::StrCat("Unknown solver type: ", solver_type));
  }
}