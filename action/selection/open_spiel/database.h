#ifndef OPEN_SPIEL_GAMES_DATABASE_H_
#define OPEN_SPIEL_GAMES_DATABASE_H_

#include <pqxx/pqxx>

#include "open_spiel/spiel.h"

namespace httplib {
class Client;
}

namespace open_spiel {
namespace database {

struct ForecastedQuery {
  std::string sql_;
  long long num_arrivals_;
  ForecastedQuery(std::string query, long long num_arrivals) : sql_(std::move(query)), num_arrivals_(num_arrivals) {}
};

class Forecast {
 public:
  explicit Forecast(const std::string &forecast_path);
  // TODO(WAN): GetForecastWorkload should factor in time.
  [[nodiscard]] const std::vector<ForecastedQuery> &GetForecastWorkload() const { return workload_; }

 private:
  std::vector<ForecastedQuery> workload_;
};

struct TuningAction {
  std::string sql_;
  explicit TuningAction(std::string sql) : sql_(std::move(sql)) {}
};

class Tuner {
 public:
  explicit Tuner(const std::string &actions_path);
  const std::vector<TuningAction> &GetTuningActions() const { return actions_; }
  const TuningAction &GetAction(size_t index) const { return actions_.at(index); }

 private:
  std::vector<TuningAction> actions_;
};

class DatabaseGame;

class DatabaseState : public State {
 public:
  explicit DatabaseState(std::shared_ptr<const Game> game);
  DatabaseState(const DatabaseState &) = default;

  Player CurrentPlayer() const override { return IsTerminal() ? kTerminalPlayerId : current_player_; }
  std::string ActionToString(Player player, Action action_id) const override;
  std::string ToString() const override;
  bool IsTerminal() const override;
  std::vector<double> Returns() const override;
  std::string InformationStateString(Player player) const override;
  std::string ObservationString(Player player) const override;
  std::unique_ptr<State> Clone() const override;
  void UndoAction(Player player, Action move) override;
  std::vector<Action> LegalActions() const override;

  std::string GetActionSQL(Action action_id) const;

 protected:
  void DoApplyAction(Action move) override;

  template<class T>
  void ApplyHistory(T &txn, bool use_hypopg) const;

  std::string GetAppliedStateRepresentation() const;

  // Gets the actual runtime cost with EXPLAIN(ANALYZE, BUFFERS) of executing query
  // under the given transaction. The function returns a pair of <success, time>.
  template<class T>
  std::pair<bool, double> GetExplainAnalyzeCostUs(T &txn, const std::string &query) const;

 private:
  std::shared_ptr<const DatabaseGame> game_;
  std::set<Action> actions_applied_;
  int num_tuning_actions_applied_ = 0;

  Player current_player_ = 0;
  bool finished_ = false;
};

class DatabaseGame : public Game {
 public:
  explicit DatabaseGame(const GameParameters &params);
  int NumDistinctActions() const override { return 500; }
  std::unique_ptr<State> NewInitialState() const override {
    return std::unique_ptr<State>(new DatabaseState(shared_from_this()));
  }
  int NumPlayers() const override { return 1; }
  double UtilitySum() const override { return 0; }

  // TODO(WAN): Tighter bounds on the utility could lead to pruning.
  double MinUtility() const override { return std::numeric_limits<double>::min(); }
  double MaxUtility() const override { return std::numeric_limits<double>::max(); }

  int MaxGameLength() const override { return MaxTuningActions(); }
  int MaxTuningActions() const { return max_tuning_actions_; }

  // TODO(WAN): Until <experimental/memory> gets experimental::observer_ptr in, raw pointers denote non-ownership.
  Forecast *GetForecast() const { return forecast_.get(); }
  Tuner *GetTuner() const { return tuner_.get(); }

  // Knobs and configuration.
  const std::string &GetDatabaseConnectionString() const { return db_conn_string_; }
  bool UseHypoPG() const { return use_hypopg_; }
  bool UseMicroservice() const { return use_microservice_; }
  bool RecordPredictions() const { return record_predictions_; }

  httplib::Client *GetMicroserviceClient() const { return microservice_client_.get(); }

 private:
  std::string db_conn_string_;
  std::unique_ptr<Forecast> forecast_;
  std::unique_ptr<Tuner> tuner_;
  int max_tuning_actions_;
  bool use_hypopg_;
  bool use_microservice_;
  bool record_predictions_;

  // use_microservice_ options.
  std::unique_ptr<httplib::Client> microservice_client_;
};

}  // namespace database
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_DATABASE_H_
