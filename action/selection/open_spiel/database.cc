#include "database.h"

#include <fstream>
#include <iostream>
#include <memory>
#include <pqxx/pqxx>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>

#include "httplib.h"
#include "rapidjson/document.h"
#include "rapidjson/prettywriter.h"
#include "rapidjson/stringbuffer.h"

namespace open_spiel {
namespace database {

namespace {

// clang-format off
// Facts about the game.
const GameType kGameType{
    /*short_name=*/"database_game",
    /*long_name=*/"Database Game",
    GameType::Dynamics::kSequential,
    GameType::ChanceMode::kDeterministic,
    GameType::Information::kPerfectInformation,
    GameType::Utility::kZeroSum,
    GameType::RewardModel::kTerminal,
    /*max_num_players=*/1,
    /*min_num_players=*/1,
    /*provides_information_state_string=*/true,
    /*provides_information_state_tensor=*/false,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/false,
    /*parameter_specification=*/{
        {"db_conn_string", GameParameter("")},
        {"forecast_path", GameParameter("")},
        {"actions_path", GameParameter("")},
        {"max_tuning_actions", GameParameter(1)},
        {"use_hypopg", GameParameter(true)},
        {"use_microservice", GameParameter(false)},
        {"record_predictions", GameParameter(false)},
    }
};
// clang-format on

std::shared_ptr<const Game> Factory(const GameParameters &params) {
  return std::shared_ptr<const Game>(new DatabaseGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);
}  // namespace

namespace {

//-- ExplainCost

struct ExplainCost {
 public:
  ExplainCost(pqxx::result *rset) {
    for (const auto &r : *rset) {
      for (const auto &f : r) {
        std::smatch matches;
        std::string current_str{pqxx::to_string(f)};

        const std::regex REGEXP{".*\\(cost=(\\d+\\.?\\d+)\\.\\.(\\d+\\.?\\d+) rows=(\\d+) width=(\\d+)\\)"};
        if (std::regex_match(current_str, matches, REGEXP)) {
          startup_cost_ = std::stod(matches[1].str());
          total_cost_ = std::stod(matches[2].str());
          num_rows_ = std::stol(matches[3].str());
          width_ = std::stol(matches[4].str());
          return;
        }
      }
    }
    throw std::runtime_error("ExplainCost: Could not parse pqxx::result result set.");
  }

  friend std::ostream &operator<<(std::ostream &os, const ExplainCost &cost) {
    os << "[EC(";
    os << cost.startup_cost_ << ',';
    os << cost.total_cost_ << ',';
    os << cost.num_rows_ << ',';
    os << cost.width_ << ")]";
    return os;
  }

 public:
  double startup_cost_ = -1;
  double total_cost_ = -1;
  long num_rows_ = -1;
  long width_ = -1;
};

//-- ExplainMicroserviceCost

struct ExplainMicroserviceCost {
 public:
  explicit ExplainMicroserviceCost(httplib::Client *client, pqxx::result *rset, bool record_predictions) {
    inference_doc_.SetArray();

    for (const auto &r : *rset) {
      for (const auto &f : r) {
        // Parse the JSON result from EXPLAIN (FORMAT JSON).
        std::string json_str{pqxx::to_string(f)};
        rapidjson::Document doc;
        doc.Parse(json_str.c_str());

        // Augment the JSON tree by computing additional properties, e.g., differencing.
        for (auto &plan_node : doc.GetArray()) {
          if (plan_node.HasMember("Tscout")) {
            rapidjson::Document::Object &&plan_obj = plan_node["Tscout"].GetObject();
            Augment(doc, plan_obj);
          }
        }

        // Accumulate the cost of each JSON node into model_cost_.
        for (auto &plan_node : doc.GetArray()) {
          if (plan_node.HasMember("Tscout")) {
            rapidjson::Document::Object &&plan_obj = plan_node["Tscout"].GetObject();
            PrepareCostRequest(inference_doc_, plan_obj);
            SendCostRequest(client, inference_doc_, record_predictions);
          }
        }
      }
    }
  }

  friend std::ostream &operator<<(std::ostream &os, const ExplainMicroserviceCost &cost) {
    os << "[EWSC(" << cost.model_cost_ << ")]";
    return os;
  }

 private:
  void Augment(rapidjson::Document &doc, rapidjson::Document::Object &node) {
    // Currently, the only augmentation done is differencing.
    // Differencing refers to isolating a node's contribution to a feature by subtracting away children contributions.
    // For example, for this plan:
    //  LockRows  (cost=0.29..8.32 rows=1 width=297)
    //   ->  Index Scan using stock_pkey on stock  (cost=0.29..8.31 rows=1 width=297)
    //         Index Cond: ((s_w_id = 1) AND (s_i_id = 69279))
    // LockRows has a differenced startup cost of 0 and differenced total cost of 0.01.

    // Compute the contribution of the children.
    double children_startup_cost = 0;
    double children_total_cost = 0;
    if (node.HasMember("Plans")) {
      for (auto &child_node : node["Plans"].GetArray()) {
        rapidjson::Document::Object &&child = child_node.GetObject();
        // Get the cost before we call Augment() on the child.
        children_startup_cost += GetValue(child, "startup_cost").GetDouble();
        children_total_cost += GetValue(child, "total_cost").GetDouble();
        Augment(doc, child);
      }
    }
    // Compute the diffed features.
    double diffed_startup_cost = GetValue(node, "startup_cost").GetDouble() - children_startup_cost;
    double diffed_total_cost = GetValue(node, "total_cost").GetDouble() - children_total_cost;
    // Add the diffed features back to the JSON object.
    GetValue(node, "startup_cost") = rapidjson::Value(diffed_startup_cost);
    GetValue(node, "total_cost") = rapidjson::Value(diffed_total_cost);
  }

  void PrepareCostRequest(rapidjson::Document &target, rapidjson::Document::Object &node) {
    rapidjson::Value node_element;
    node_element.SetObject();

    rapidjson::Value features;
    features.SetObject();

    for (rapidjson::Value::MemberIterator it = node.MemberBegin(); it != node.MemberEnd(); it++) {
      // Pass all features provided by Hutch directly to model inference only if
      // the attribute value is a numeric scalar.
      if (!it->value.IsObject() && !it->value.IsArray() && !it->value.IsString()) {
        // Explicitly invoke rapidjson::Value() copy constructor so we aren't moving
        // elements from one node to another.
        features.AddMember(rapidjson::Value(it->name, target.GetAllocator()),
                           rapidjson::Value(it->value, target.GetAllocator()), target.GetAllocator());
      }
    }

    // Determine which behavior model type to use.
    node_element.AddMember("model_type", "rf", target.GetAllocator());
    // Just kidding. No choice allowed, random forests only.
    // TODO(WAN): More seriously, we should expose this as a parameter.

    // Determine which behavior model this is.
    // TODO(wz2): Optionally also attach `node` so we can match predictions back to the plan tree.
    auto node_member = node.FindMember("node_type");
    assert(node_member != node.MemberEnd() && "Expected to find node_type member.");
    node_element.AddMember("ou_type", rapidjson::Value(node_member->value, target.GetAllocator()),
                           target.GetAllocator());
    node_element.AddMember("features", features, target.GetAllocator());
    target.PushBack(node_element, target.GetAllocator());

    // Repeat the costing process for all children.
    if (node.HasMember("Plans")) {
      for (auto &child_node : node["Plans"].GetArray()) {
        rapidjson::Document::Object &&child = child_node.GetObject();
        PrepareCostRequest(target, child);
      }
    }
  }

  void SendCostRequest(httplib::Client *client, rapidjson::Document &doc, bool record_predictions) {
    rapidjson::StringBuffer buffer;
    {
      // Serialize the document into the buffer so we can attach it as JSON to the
      // HTTP POST request to the microservice.
      rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
      doc.Accept(writer);
    }

    std::string url = "/batch_infer";
    if (auto res = client->Post(url.c_str(), buffer.GetString(), buffer.GetSize(), "application/json")) {
      // If it looks like we got a JSON response, we're probably fine.
      if (res->status == 200 && absl::StartsWith(res->body, "[")) {
        // Parse the JSON response and add the relevant attributes to our cost.
        rapidjson::Document doc;
        doc.Parse(res->body.c_str());
        for (rapidjson::Value::ConstValueIterator it = doc.Begin(); it != doc.End(); ++it) {
          if (it->IsObject() && it->HasMember("elapsed_us")) {
            model_cost_ += (*it)["elapsed_us"].GetDouble();
          } else {
            // We can't write res->body since that is the entire response body. Instead,
            // we just take this particular element, serialize it, and output it.
            buffer.Clear();
            rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
            it->Accept(writer);

            std::cerr << absl::StrCat("ERROR bad result: status ", res->status, " url ", url, " body ",
                                      buffer.GetString())
                      << std::endl;
          }
        }
      } else {
        std::cerr << absl::StrCat("ERROR unknown: status ", res->status, " url ", url, " body ", res->body)
                  << std::endl;
      }

      if (record_predictions) {
        // If we're recording results, then append this particular inference result to the accumulator.
        // We'll be able to identify this failure since the model_cost_ will show a low prediction.
        inference_results_ << (res->body.c_str()) << std::endl;
      }
    } else {
      auto err = res.error();
      std::cerr << absl::StrCat("ERROR bad URL: status ", err, " url /batch_infer") << std::endl;
    }
  }

  bool EndsWith(std::string const &fullString, std::string const &suffix) {
    if (fullString.length() >= suffix.length()) {
      return (0 == fullString.compare(fullString.length() - suffix.length(), suffix.length(), suffix));
    }

    return false;
  }

  rapidjson::Value &GetValue(rapidjson::Value &node, std::string key) {
    for (rapidjson::Value::MemberIterator itr = node.MemberBegin(); itr != node.MemberEnd(); itr++) {
      if (EndsWith(itr->name.GetString(), key)) {
        return itr->value;
      }
    }

    throw std::runtime_error(absl::StrCat("Could not fetch key: ", key, "from document"));
  }

 public:
  // The model cost is aggregated by making network requests.
  double model_cost_ = 0;

  // std::stringstream used to concatenate together the results from each inference requeired
  // to process the entire EXPLAIN query tree. Inference results are only gathered when
  // record_predictions is enabled.
  std::stringstream inference_results_;

  // The JSON document that is used to construct the request to the microservice. This is used
  // so we can request inference for the entire EXPLAIN tree in a single request.
  // We observed 2x speedup on dev4 (1 hour -> 30 minutes) by doing so.
  rapidjson::Document inference_doc_;
};

//-- ExplainAnalyzeCost

struct ExplainAnalyzeCost {
 public:
  ExplainAnalyzeCost(pqxx::result *rset) {
    bool done[3] = {false, false, false};
    for (const auto &r : *rset) {
      for (const auto &f : r) {
        std::smatch matches;
        std::string current_str{pqxx::to_string(f)};

        if (!done[0]) {
          const std::regex REGEXP{
              ".*\\(cost=(\\d+\\.?\\d+)\\.\\.(\\d+\\.?\\d+) rows=(\\d+) width=(\\d+)\\).*\\"
              "(actual time=(\\d+\\.?\\d+)\\.\\.(\\d+\\.?\\d+) rows=(\\d+) loops=(\\d+)\\)"};
          if (std::regex_match(current_str, matches, REGEXP)) {
            startup_cost_ = std::stod(matches[1].str());
            total_cost_ = std::stod(matches[2].str());
            num_rows_ = std::stol(matches[3].str());
            width_ = std::stol(matches[4].str());
            actual_startup_time_ms_ = std::stod(matches[5].str());
            actual_total_time_ms_ = std::stod(matches[6].str());
            actual_num_rows_ = std::stol(matches[7].str());
            actual_loops_ = std::stol(matches[8].str());
            done[0] = true;
            continue;
          }
        }

        if (!done[1]) {
          const std::regex REGEXP_PLAN{"Planning Time: (\\d+\\.?\\d+) ms"};
          if (std::regex_match(current_str, matches, REGEXP_PLAN)) {
            actual_planning_time_ms_ = std::stod(matches[1].str());
            done[1] = true;
            continue;
          }
        }

        if (!done[2]) {
          const std::regex REGEXP_EXEC{"Execution Time: (\\d+\\.?\\d+) ms"};
          if (std::regex_match(current_str, matches, REGEXP_EXEC)) {
            actual_execution_time_ms_ = std::stod(matches[1].str());
            done[2] = true;
            return;
          }
        }
      }
    }
    throw std::runtime_error("ExplainAnalyzeCost: Could not parse pqxx::result result set.");
  }

  friend std::ostream &operator<<(std::ostream &os, const ExplainAnalyzeCost &cost) {
    os << "[EAC(";
    os << cost.startup_cost_ << ',';
    os << cost.total_cost_ << ',';
    os << cost.num_rows_ << ',';
    os << cost.width_ << ',';
    os << cost.actual_startup_time_ms_ << ',';
    os << cost.actual_total_time_ms_ << ',';
    os << cost.actual_num_rows_ << ',';
    os << cost.actual_loops_ << ',';
    os << cost.actual_planning_time_ms_ << ',';
    os << cost.actual_execution_time_ms_ << ")]";
    return os;
  }

 public:
  double startup_cost_ = -1;
  double total_cost_ = -1;
  long num_rows_ = -1;
  long width_ = -1;
  double actual_startup_time_ms_ = -1;
  double actual_total_time_ms_ = -1;
  long actual_num_rows_ = -1;
  long actual_loops_ = -1;
  double actual_planning_time_ms_ = -1;
  double actual_execution_time_ms_ = -1;
};

}  // namespace

//-- Forecast

Forecast::Forecast(const std::string &forecast_path) {
  // Parse the forecast CSV file.
  std::ifstream forecast_csv(forecast_path);
  if (!forecast_csv.is_open() || !forecast_csv.good()) {
    throw std::runtime_error(absl::StrCat("Couldn't read forecast file: ", forecast_path));
  }

  std::cerr << "Forecast: reading from " << forecast_path << std::endl;
  std::string line;
  long long num_queries_total = 0;
  while (std::getline(forecast_csv, line)) {
    // TODO(WAN): Fragile, but we control the output format + we may replace this code with SQL.
    // Current format: "query","count". It looks horrible because CSV quotes " with "".
    size_t split_index = line.rfind("\",\"");
    std::string query = line.substr(1, split_index - 1);
    // We have to replace the double-quotes though, otherwise we can hit length limits.
    std::regex unescape_double_quotes("\"\"");
    query = std::regex_replace(query, unescape_double_quotes, "\"");
    long long num_times = std::stoll(line.substr(split_index + 3, line.length() - (split_index + 3) - 1));
    workload_.emplace_back(query, num_times);
  }
  std::cerr << "\tRead " << workload_.size() << " distinct queries, total " << num_queries_total << "." << std::endl;
  std::cerr << std::endl;
}

//-- Tuner

Tuner::Tuner(const std::string &actions_path) {
  // Parse the actions CSV file.
  std::ifstream actions_csv(actions_path);
  if (!actions_csv.is_open() || !actions_csv.good()) {
    throw std::runtime_error(absl::StrCat("Couldn't read actions file: ", actions_path));
  }

  std::cerr << "Tuner: reading actions from " << actions_path << std::endl;
  std::string line;
  while (std::getline(actions_csv, line)) {
    // TODO(WAN): This format may change and we need documentation.
    std::string sql = line;
    actions_.emplace_back(sql);
  }
  std::cerr << "\tRead " << actions_.size() << " actions." << std::endl;
  std::cerr << std::endl;
}

//-- DatabaseState

DatabaseState::DatabaseState(std::shared_ptr<const Game> game)
    : State(game), game_(std::dynamic_pointer_cast<const DatabaseGame>(game)) {}

void DatabaseState::DoApplyAction(Action move) {
  // Check: Action shouldn't have been applied yet.
  SPIEL_CHECK_TRUE(actions_applied_.find(move) == actions_applied_.end());

  // Track which actions have been applied.
  actions_applied_.emplace(move);
  ++num_tuning_actions_applied_;

  // Stop the game after applying N tuning actions.
  if (num_tuning_actions_applied_ >= game_->MaxTuningActions()) {
    finished_ = true;
  }
}

void DatabaseState::UndoAction(Player player, Action move) {
  // OpenSpiel.
  current_player_ = player;
  history_.pop_back();
  --move_number_;

  // Database game.
  SPIEL_CHECK_TRUE(actions_applied_.find(move) != actions_applied_.end());
  actions_applied_.erase(move);
  --num_tuning_actions_applied_;
}

std::vector<Action> DatabaseState::LegalActions() const {
  // If the game is over, there are no more moves.
  if (IsTerminal()) {
    return {};
  }

  // Otherwise, the valid actions are those which have not already been applied.
  std::vector<Action> moves;
  {
    size_t num_tuning_actions = game_->GetTuner()->GetTuningActions().size();
    moves.reserve(num_tuning_actions);
    for (Action action = 0; action < num_tuning_actions; ++action) {
      if (actions_applied_.find(action) == actions_applied_.end()) {
        moves.emplace_back(action);
      }
    }
  }

  return moves;
}

std::string DatabaseState::GetActionSQL(Action action_id) const {
  const std::string &sql = game_->GetTuner()->GetTuningActions().at(action_id).sql_;
  return sql;
}

std::string DatabaseState::ActionToString(Player player, Action action_id) const {
  return absl::StrCat("Action[", "player=", player, ",action=", action_id, "]");
}

std::string DatabaseState::ToString() const {
  std::ostringstream os;
  os << "History[";
  if (!history_.empty()) {
    os << GetActionSQL(history_.at(0).action);
  }
  for (size_t i = 1; i < history_.size(); ++i) {
    os << " ";
    os << GetActionSQL(history_.at(i).action);
  }
  os << "]";
  return os.str();
}

bool DatabaseState::IsTerminal() const { return finished_; }

void DatabaseState::ApplyHistory(pqxx::dbtransaction &txn) const { CustomApplyHistory(txn, game_->UseHypoPG()); }

void DatabaseState::CustomApplyHistory(pqxx::dbtransaction &txn, bool use_hypopg) const {
  // Apply all the tuning actions. This assumes that actions take zero time.
  for (const auto &player_action : history_) {
    // Apply the tuning action.
    {
      const TuningAction &action = game_->GetTuner()->GetAction(player_action.action);
      std::string action_sql = action.sql_;
      // If hypopg is enabled and this is a CREATE INDEX action, we need to modify the action SQL.
      // TODO(WAN): Be more robust about cReaTE inDeX.
      bool is_create_index = action_sql.find("CREATE INDEX") == 0 || action_sql.find("create index") == 0;
      if (use_hypopg && is_create_index) {
        // hypopg_create_index() is ok with the action_sql ending with a semicolon.
        action_sql = absl::StrCat("SELECT * FROM hypopg_create_index('", action_sql, "');");
      }
      pqxx::result rset{txn.exec(action_sql)};
    }
  }
}

struct ExplainAnalyzeCostResult DatabaseState::GetExplainAnalyzeCostUs(pqxx::dbtransaction &txn,
                                                                       const std::string &query) const {
  // By default, EXPLAIN() returns timestamps as milliseconds. However, model forecasting for
  // elapsed time returns timestamps as microseconds instead. As such, this function provides
  // the explain cost (elapsed time) in microseconds to be consistent with the inference.
  std::string explain = absl::StrCat("EXPLAIN (ANALYZE, BUFFERS) ", query);

  try {
    pqxx::result rset{txn.exec(explain)};
    std::string explain = absl::StrCat("EXPLAIN (ANALYZE, BUFFERS) ", query);
    ExplainAnalyzeCost cost{&rset};
    return ExplainAnalyzeCostResult{true, cost.actual_total_time_ms_ * 1000};
  } catch (const pqxx::integrity_constraint_violation &e) {
    // EXPLAIN ANALYZE will execute the query that it is provided as part of the workload forecast.
    // However, this query may abort due to failing integrity constraint check(s) such as uniqueness.
    // We currently do not model nor handle aborts, therefore all aborted queries will be treated as
    // if they had zero cost.
    //
    // TODO(wz2): Need a graceful way of identifying and possibly modeling aborts.
    std::cerr << absl::StrCat("ERROR while executing query ", explain, " ", e.what()) << std::endl;
    return ExplainAnalyzeCostResult{false, 0.0};
  }
}

std::vector<double> DatabaseState::Returns() const {
  // Compute the final reward of this game trajectory.
  // TODO(WAN): Narrowing and overflow issues for cost computation?
  double total_cost = 0;

  // Game parameters.
  const std::string &db_conn_string = game_->GetDatabaseConnectionString();
  bool use_hypopg = game_->UseHypoPG();
  bool use_microservice = game_->UseMicroservice();
  bool record_predictions = game_->RecordPredictions();
  // Non-owning pointers to Forecast and Tuner.
  Forecast *forecast = game_->GetForecast();
  Tuner *tuner = game_->GetTuner();

  // Create a connection to the DBMS.
  pqxx::connection conn(db_conn_string);
  pqxx::work txn(conn);

  // If HypoPG is enabled, reset HypoPG state.
  if (use_hypopg) {
    // hypopg_reset() removes all hypothetical indexes.
    txn.exec("select hypopg_reset();");
  }

  httplib::Client *client;
  if (use_microservice) {
    client = game_->GetMicroserviceClient();
    SPIEL_CHECK_NE(client, nullptr);
  }

  // Simulate the game trajectory.
  {
    ApplyHistory(txn);

    // TODO(WAN): Faking out the forecast time should go here.
    //  e.g., pass the current time as an input into GetForecastWorkload(),
    //  and move this block of code into the above for (history) loop.
    //  Right now, the workload only appears at the very end after all actions are applied.
    const std::vector<ForecastedQuery> &workload = forecast->GetForecastWorkload();

    // TOOD(WAN): If you find yourself needing to declare more flag-specific variables here,
    // refactor this into an `EvaluateWorkloadCost` function that calls separate costing
    // functions.
    //
    // This vector is used to record predictions. The vector (if valid) corresponds 1:1
    // with the previous workload vector. The vector stores a pair of the query's
    // predicted runtime cost and the debug info generated by the inference process.
    std::vector<std::pair<double, std::string>> predicted_workloads;

    // Evaluate the cost of the forecasted queries.
    {
      for (const auto &work : workload) {
        std::string query = work.sql_;
        if (use_hypopg && !use_microservice) {
          query = absl::StrCat("EXPLAIN ", query);
          pqxx::result rset{txn.exec(query)};
          ExplainCost cost{&rset};
          total_cost += cost.total_cost_ * work.num_arrivals_;
        } else if (use_hypopg && use_microservice) {
          query = absl::StrCat("EXPLAIN (FORMAT tscout) ", query);
          pqxx::result rset{txn.exec(query)};
          ExplainMicroserviceCost cost{client, &rset, record_predictions};
          total_cost += cost.model_cost_ * work.num_arrivals_;
          if (record_predictions) {
            predicted_workloads.emplace_back(cost.model_cost_, cost.inference_results_.str());
          }
        } else {
          total_cost += GetExplainAnalyzeCostUs(txn, query).cost_ * work.num_arrivals_;
        }
      }
    }

    // This is conditioned on use_hypopg only because we only support recording predictions
    // in the use_hypopg and use_microservice code-path.
    if (use_hypopg && use_microservice && record_predictions) {
      // Remove all hypothetical indexes.
      txn.exec("select hypopg_reset();");

      // Start a sub-transaction. When this sub-transaction goes away, all of the actions
      // that we apply (which we assume are transactional) will be automatically rolled back.
      pqxx::subtransaction subtxn_history(txn);
      CustomApplyHistory(subtxn_history, false);
      std::string action_state = ToString();

      // This is the sub-transaction that we use to execute the workload. This should see the
      // the changes introduced by subtxn_history. subtxn is used to execute the workload.
      // If we encounter an error, we don't need to redo all the work done by subtxn_history.
      pqxx::subtransaction subtxn(subtxn_history);

      SPIEL_CHECK_EQ(workload.size(), predicted_workloads.size());
      rapidjson::Document results;
      results.SetArray();

      for (size_t idx = 0; idx < workload.size(); idx++) {
        // Get the true cost of executing this query with the current configuration.
        struct ExplainAnalyzeCostResult true_cost = GetExplainAnalyzeCostUs(subtxn, workload[idx].sql_);
        if (!true_cost.cost_valid_) {
          // In this case, the subtransaction has failed (e.g., we hit an integrity key
          // violation). For this case, we need to restart the subtransaction.
          subtxn.~subtransaction();
          new (&subtxn) pqxx::subtransaction(subtxn_history);
        }

        rapidjson::Value result;
        result.SetObject();
        result.AddMember("predicted_cost", predicted_workloads[idx].first, results.GetAllocator());
        result.AddMember("true_cost_valid", true_cost.cost_valid_, results.GetAllocator());
        result.AddMember("true_cost", true_cost.cost_, results.GetAllocator());
        result.AddMember("query", rapidjson::StringRef(workload[idx].sql_.c_str()), results.GetAllocator());
        result.AddMember("predicted_results", rapidjson::StringRef(predicted_workloads[idx].second.c_str()),
                         results.GetAllocator());
        result.AddMember("action_state", rapidjson::StringRef(action_state.c_str()), results.GetAllocator());
        results.PushBack(result, results.GetAllocator());
      }

      rapidjson::StringBuffer buffer;
      rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
      results.Accept(writer);
      if (auto res = client->Post("/prediction_results", buffer.GetString(), buffer.GetSize(), "application/json")) {
        if (res->status != 200) {
          std::cerr << absl::StrCat("ERROR unknown: status ", res->status, " body ", res->body) << std::endl;
        }
      }
    }
  }

  // The reward is the negative of the total cost because we want to pay less total cost.
  return {-total_cost};
}

std::string DatabaseState::InformationStateString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return HistoryString();
}

std::string DatabaseState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return ToString();
}

std::unique_ptr<State> DatabaseState::Clone() const { return std::unique_ptr<State>(new DatabaseState(*this)); }

//-- DatabaseGame

DatabaseGame::DatabaseGame(const GameParameters &params) : Game(kGameType, params) {
  db_conn_string_ = params.at("db_conn_string").string_value();
  forecast_ = std::make_unique<Forecast>(params.at("forecast_path").string_value());
  tuner_ = std::make_unique<Tuner>(params.at("actions_path").string_value());
  max_tuning_actions_ = params.at("max_tuning_actions").int_value();
  use_hypopg_ = params.at("use_hypopg").bool_value();
  use_microservice_ = params.at("use_microservice").bool_value();
  record_predictions_ = params.at("record_predictions").bool_value();

  // Microservice currently assumes HypoPG. There is no reason this has to be the case.
  SPIEL_CHECK_TRUE(!use_microservice_ || (use_microservice_ && use_hypopg_));

  // Recording currently requires microservice. This is because data is recorded to the microservice.
  SPIEL_CHECK_TRUE(!record_predictions_ || (record_predictions_ && use_microservice_));

  if (use_microservice_) {
    // 127.0.0.1:5000 is the Flask default.
    microservice_client_ = std::make_unique<httplib::Client>("127.0.0.1", 5000);
    if (record_predictions_) {
      microservice_client_->Delete("/prediction_results");
    }
  }
}

}  // namespace database
}  // namespace open_spiel
